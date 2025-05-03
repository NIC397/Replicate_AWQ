import torch
from torch import nn
import gc
from awq_utils import get_op_name, get_op_by_name
from awq_utils import quantize_weight
import copy
@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)

@torch.no_grad()
def auto_scale_block(module, module_kwargs, num_bits, group_size, input_feat, s_val=None):
    def w_quantize_func(p):
        return quantize_weight(
            p,
            num_bits=num_bits,
            group_size=group_size,
        ).detach()

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    def optimal_scales(block, linears2scale: list, x, kwargs={},s_val=None):
        x = x.to(next(block.parameters()).device)
        if s_val is not None:
            scales = torch.full((x.shape[-1],), 2, dtype=x.dtype, device=x.device)
            return scales.view(-1).detach()
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        x_max = get_act_scale(x)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20

        original_state = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
           
            out = block(x, **kwargs)
    
            if isinstance(out, tuple):
                out = out[0]

            loss = ((org_out - out).float().pow(2).mean().item())  # float prevents overflow
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(original_state)
        if best_ratio == -1:
            raise Exception
        # print(best_ratio)
        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()


    def get_scales(prev_op, layers, inp, inspect_module = None, kwargs={}, s_val=None):
        # TODO - complete
        if inspect_module is None:
            assert len(layers) == 1
            inspect_module = layers[0]
    
        scales = optimal_scales(inspect_module, layers, inp, kwargs, s_val)
        scales = scales.detach().cpu()
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
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

# apply_awq
def apply_awq_scaling(model, awq_results):
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])



def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm,)):
            scale_ln_fcs(prev_op, layers, scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device).to(inp.dtype))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()

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


# weight quantization
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, num_bits, group_size, n_grid=20, max_shrink=0.5, n_sample_token=512
):
  
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = quantize_weight(cur_w, num_bits=num_bits, group_size=group_size)
            cur_out = (input_feat * q_w).sum(dim=-1)

            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    gc.collect()
    torch.cuda.empty_cache()
    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module, num_bits, group_size, input_feat):
    named_linears = {
        name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)
    }

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        max_val = auto_clip_layer(
            named_linears[name].weight, input_feat[name], num_bits=num_bits, group_size=group_size
        )
        clip_list.append((name, max_val))
        named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()
    gc.collect()
    torch.cuda.empty_cache()
