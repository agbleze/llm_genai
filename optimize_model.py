
#%%
import torch
torch.cuda.init()
from transformers import AutoModelForCausalLM, AutoTokenizer
import tensorrt as trt
#%%
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

#%%
model.eval()



# %%

text = "Explain the information theory"
inputs = tokenizer(text, return_tensors="pt")

# %%
torch.onnx.export(model, 
                  args=({
                      "input_ids": inputs["input_ids"],
                      "attention_mask": inputs["attention_mask"],
                      "past_key_values": None,
                      "use_cache": False,
                      "return_dict": False
                  }),
                  #(inputs["input_ids"], inputs["attention_mask"]),
                  f="gpt2.onnx",
                  input_names=["input_ids", "attention_mask"],
                  output_names=["logits"],
                  opset_version=17,
                  dynamic_axes={"input_ids": {0: "batch_size", 1: "seq"},
                                "attention_mask":  {0: "batch", 1: "seq"},
                                "logits": {0: "batch", 1: "seq"}
                                },
                #   kwargs={
                #       "input_ids": inputs["input_ids"],
                #       "attention_mask": inputs["attention_mask"],
                #       "past_key_values": None,
                #       "use_cache": False,
                #       "return_dict": False
                #   }
                  )


#%%

# %% 3. PREPARE DUMMY RUN & EXPORT TO ONNX (PyTorch 2.4+ Compliant)
text = "Explain the information theory"
inputs = tokenizer(text, return_tensors="pt")

# Move dummy inputs to the GPU to match the model context
input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()
model = model.cuda()

print("Exporting model to ONNX...")
torch.onnx.export(
    model, 
    # Positional inputs go in args as a flat tuple
    args=(input_ids, attention_mask),
    f="gpt2.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    opset_version=17,
    # Non-tensor structural configuration flags go in kwargs
    kwargs={
        "past_key_values": None,
        "use_cache": False,
        "return_dict": False
    },
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq"},
        "attention_mask": {0: "batch_size", 1: "seq"},
        "logits": {0: "batch_size", 1: "seq"}
    }
)
print("ONNX export complete.")



# %%
def build_trt_engine(onnx_path, fp16=True, max_workspace_size_gb=4):
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print("ONNX parse error: ", parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")
        
    config = builder.create_builder_config()
    #config.max_workspace_size = max_workspace_size_gb * (1024 ** 3)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size_gb * (1024 ** 3))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    #engine = builder.build_engine(network, config)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    engine_path = onnx_path.replace(".onnx", ".engine")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"TensorRT engine saved to {engine_path}")
    return serialized_engine


#%%
engine = build_trt_engine("gpt2.onnx", fp16=True)




# %%
!python3 -c "import torch; print('Torch CUDA:', torch.version.cuda)"
!python3 -c "import tensorrt; print('TRT Version:', tensorrt.__version__)"

# %%
