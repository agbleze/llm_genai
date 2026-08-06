
#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#%%
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
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

# %%
