import torch
from torch import nn
import awq_inference_engine

def round_up_to_multiple(value, factor):
    return (value + factor - 1) // factor

def compute_group_width(dim, grp_size=128, pack_factor=8):
    adjusted_width = round_up_to_multiple(dim // grp_size, pack_factor)
    return adjusted_width

def quantize_and_pack_weights(raw_weights, row_block, col_block):
    M = raw_weights.shape[0]
    N = raw_weights.shape[1]

    kernel_data = raw_weights.cpu().numpy().reshape(M, N // 32, 32)
    kernel_data = kernel_data.reshape(M, N // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    kernel_data = kernel_data.reshape(M, N // 32, 32)

    kernel_data = kernel_data.reshape(M, N // 32, 4, 8)
    kernel_data = kernel_data.reshape(M, N // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    kernel_data = kernel_data.reshape(M, N)

    kernel_data = kernel_data.reshape(M // row_block, row_block, N // col_block, col_block)
    kernel_data = kernel_data.transpose(0, 2, 1, 3)
    kernel_data = kernel_data.reshape(M // row_block, N // col_block, col_block, row_block)

    kernel_data = (
        kernel_data[..., 0]
        | (kernel_data[..., 1] << 4)
        | (kernel_data[..., 2] << 8)
        | (kernel_data[..., 3] << 12)
    )

    kernel_data = kernel_data.reshape(M // row_block, N)
    packed_tensor = torch.tensor(kernel_data.astype("int16")).to(raw_weights.device).contiguous()
    return packed_tensor

class QuantizedLinearLayer(nn.Module):
    def __init__(self, bit_width, grp_sz, input_dim, output_dim, has_bias, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bit_width = bit_width
        self.grp_sz = grp_sz
        self.tile_k = 8
        self.row_interleave = 4

        per_pack = 32 // self.bit_width
        half_pack = 16 // self.bit_width

        self.register_buffer(
            "packed_weights",
            torch.zeros(
                (output_dim // self.row_interleave, input_dim // half_pack * self.row_interleave),
                dtype=torch.int16,
                device=device,
            ),
        )

        self.register_buffer(
            "scaling_factors",
            torch.zeros(
                (compute_group_width(input_dim, self.grp_sz) * per_pack, output_dim),
                dtype=torch.float16,
                device=device,
            ),
        )

        self.register_buffer(
            "adjusted_zeros",
            torch.zeros(
                (compute_group_width(input_dim, self.grp_sz) * per_pack, output_dim),
                dtype=torch.float16,
                device=device,
            ),
        )

        if has_bias:
            self.register_buffer(
                "bias_term", torch.zeros((output_dim,), dtype=torch.float16, device=device)
            )
        else:
            self.bias_term = None

    @classmethod
    def from_dense_layer(cls, dense_layer, bit_width, grp_sz, scale_mat=None, zero_mat=None):
        new_layer = cls(
            bit_width,
            grp_sz,
            dense_layer.in_features,
            dense_layer.out_features,
            dense_layer.bias is not None,
            dense_layer.weight.device,
        )

        fused_zeros = zero_mat * scale_mat
        per_pack = 32 // new_layer.bit_width

        padded_scales = torch.zeros(
            (scale_mat.shape[0], compute_group_width(dense_layer.in_features, grp_sz) * per_pack),
            dtype=torch.float16,
            device=scale_mat.device,
        )
        padded_scales[:, : scale_mat.shape[1]] = scale_mat
        new_layer.scaling_factors = padded_scales.transpose(1, 0).contiguous()

        if dense_layer.bias is not None:
            new_layer.bias_term = dense_layer.bias.clone().half()

        quantized_weights = []
        for col in range(new_layer.input_dim):
            group_id = col // grp_sz
            quantized_weights.append(
                torch.round(
                    (dense_layer.weight.data[:, col] + fused_zeros[:, group_id]) /
                    padded_scales[:, group_id]
                ).to(torch.int)[:, None]
            )

        quantized_tensor = torch.cat(quantized_weights, dim=1).to(dtype=torch.int32)
        new_layer.packed_weights = quantize_and_pack_weights(
            quantized_tensor.contiguous(), row_block=4, col_block=64
        )

        zero_mat = zero_mat.to(dtype=torch.int32)
        scaled_zeros = torch.zeros_like(padded_scales)
        scaled_zeros[:, : scale_mat.shape[1]] = -(
            padded_scales[:, : scale_mat.shape[1]] * zero_mat.to(torch.float32)
        ).to(torch.float16)
        new_layer.adjusted_zeros = scaled_zeros.transpose(1, 0).contiguous()

        return new_layer

    @torch.no_grad()
    def forward(self, input_tensor):
        if input_tensor.numel() / input_tensor.shape[-1] < 8:
            result = awq_inference_engine.gemv_forward_cuda_new(
                input_tensor,
                self.packed_weights,
                self.scaling_factors,
                self.adjusted_zeros,
                input_tensor.numel() // input_tensor.shape[-1],
                self.output_dim,
                self.input_dim,
                self.grp_sz,
            )
        else:
            result = awq_inference_engine.gemm_forward_cuda_new(
                input_tensor,
                self.packed_weights,
                self.scaling_factors,
                self.adjusted_zeros,
            )
        return result + self.bias_term if self.bias_term is not None else result

    def extra_repr(self) -> str:
        return (
            "input_dim={}, output_dim={}, bias={}, bit_width={}, group_size={}".format(
                self.input_dim,
                self.output_dim,
                self.bias_term is not None,
                self.bit_width,
                self.grp_sz,
            )
        )
