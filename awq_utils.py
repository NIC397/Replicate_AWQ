import torch
from torch import nn
import gc
from awq_core import QuantizedLinearLayer
from tqdm import tqdm

def get_blocks(model):
    return model.model.decoder.layers

def move_device(model, device_type):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device_type)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
        device_type
    )
    gc.collect()
    torch.cuda.empty_cache()

def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")

def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)
        
def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def quantize(model, num_bits, group_size):
    # TODO - complete
    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="Quantizing model",
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        for name, module in named_linears.items():
            module.cuda()
            module.weight.data, scales, zeros = quantize_tensor(
                module.weight.data, num_bits=num_bits, group_size=group_size, get_scale_zp=True
            )
            q_linear = QuantizedLinearLayer.from_dense_layer(
                module, num_bits, group_size, scales, zeros
            )
            module.cpu()
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
            torch.cuda.empty_cache()
            gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

def quantize_tensor(
    w, num_bits, group_size, get_scale_zp=False
):
    original_w_shape = w.shape

    w = w.reshape(-1, group_size)
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**num_bits - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    w = (
        torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
    ) * scales

    w = w.reshape(original_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w
    