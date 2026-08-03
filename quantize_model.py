import torch


def quantize_model(model):
    
    model.eval()
    
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    model_f32_prepared = torch.quantization.prepare(model)
    model_int8 = torch.quantization.convert(model_f32_prepared)
    return model_int8