import torch
from torch import nn
import gc
from awq_core import AWQCore
from awq_clip import apply_clip

@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)

@torch.no_grad()
def auto_scale_block(module, module_kwargs, num_bits, group_size, input_feat, s_val=None):
    def w_quantize_func(p):
        return AWQCore.AWQQuantizer.quantize_tensor(
            p,
            num_bits=num_bits,
            group_size=group_size,
        ).detach()

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def optimal_scales(block, linears2scale: list, x, kwargs={}, s_val=None):
        x = x.to(next(block.parameters()).device)

        # If fixed s_val, no need to search for it
        if s_val is not None:
            scales = torch.full((x.shape[-1],), 2, dtype=x.dtype, device=x.device)
            return scales.view(-1).detach()

        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        # Gets average activations across calibration set
        x_max = get_act_scale(x)

        best_error = float("inf")
        best_scales = None

        # Grid search from 0 to 1 in intervals of 1 / n_grid
        n_grid = 20

        original_state = {k: v.cpu() for k, v in block.state_dict().items()}
        for alpha in range(n_grid):
            alpha = alpha * 1 / n_grid
            scales = x_max.pow(alpha).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (
                (org_out - out).float().pow(2).mean().item()
            ) 

            # Find scales that lead to lowest loss
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_scales = scales
            
            # Restore block for next iteration
            block.load_state_dict(original_state)

        best_scales = best_scales.view(-1)
        return best_scales.detach()

    def get_scales(prev_op, layers, inp, inspect_module, kwargs={}, s_val=None):
        scales = optimal_scales(inspect_module, layers, inp, kwargs, s_val)
        scales = scales.detach().cpu()
        return (
            AWQCore.AWQUtils.get_op_name(module, prev_op),
            tuple([AWQCore.AWQUtils.get_op_name(module, m) for m in layers]),
            scales,
        )

    scales_list = []

    scales_list.append(
        get_scales(
            prev_op=module.self_attn_layer_norm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            inspect_module=module.self_attn,
            kwargs=module_kwargs,
            s_val=s_val
        )
    )
    scales_list.append(
        get_scales(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.out_proj],
            inp=input_feat["self_attn.out_proj"],
            inspect_module=module.self_attn.out_proj,
            s_val=s_val
        )
    )
    scales_list.append(
        get_scales(
            prev_op=module.final_layer_norm,
            layers=[module.fc1],
            inp=input_feat["fc1"],
            inspect_module=module.fc1,
            s_val=s_val
        )
    )
    scales_list.append(
        get_scales(
            prev_op=module.fc1,
            layers=[module.fc2],
            inp=input_feat["fc2"],
            inspect_module=module.fc2,
            s_val=s_val
        )
    )   
    
    gc.collect()
    torch.cuda.empty_cache()

    return scales_list

def apply_awq_scaling(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])

def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = AWQCore.AWQUtils.get_op_by_name(module, prev_op_name)
        layers = [AWQCore.AWQUtils.get_op_by_name(module, name) for name in layer_names]
        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()
        if isinstance(prev_op, nn.Linear):
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, nn.LayerNorm):
            scale_ln_fcs(prev_op, layers, scales)

        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    scales = scales.to(fc1.weight.device)
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1)) 