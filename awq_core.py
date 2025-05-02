import torch
from torch import nn
import gc
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from typing import Any

class AWQCore:
    """Core AWQ (Activation-aware Weight Quantization) implementation.
    
    This class combines the core functionality of AWQ including:
    - Linear layer implementation with AWQ
    - Quantization operations
    - Utility functions
    - Main orchestration
    """
    
    class AWQUtils:
        """Utility functions for AWQ operations."""
        
        @staticmethod
        def get_blocks(model: nn.Module) -> List[nn.Module]:
            """Get the transformer blocks from the model."""
            return model.model.decoder.layers
        
        @staticmethod
        def move_device(model: nn.Module, device_type: str) -> None:
            """Move model components to specified device."""
            model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device_type)
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device_type)
            gc.collect()
            torch.cuda.empty_cache()
        
        @staticmethod
        def append_str_prefix(x: Union[str, Tuple, List], prefix: str) -> Union[str, Tuple, List]:
            """Append prefix to string or collection of strings."""
            if isinstance(x, str):
                return prefix + x
            elif isinstance(x, tuple):
                return tuple([AWQCore.AWQUtils.append_str_prefix(y, prefix) for y in x])
            elif isinstance(x, list):
                return [AWQCore.AWQUtils.append_str_prefix(y, prefix) for y in x]
            else:
                return x
        
        @staticmethod
        def get_op_name(module: nn.Module, op: nn.Module) -> str:
            """Get the name of an operation relative to the module."""
            for name, m in module.named_modules():
                if m is op:
                    return name
            raise ValueError(f"Cannot find op {op} in module {module}")
        
        @staticmethod
        def set_op_by_name(layer: nn.Module, name: str, new_module: nn.Module) -> None:
            """Set a module by its name in the layer."""
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
        
        @staticmethod
        def get_op_by_name(module: nn.Module, op_name: str) -> nn.Module:
            """Get an operation by its name relative to the module."""
            for name, m in module.named_modules():
                if name == op_name:
                    return m
            raise ValueError(f"Cannot find op {op_name} in module {module}")
        
        @staticmethod
        def get_named_linears(module: nn.Module) -> Dict[str, nn.Linear]:
            """Get all linear layers in the module with their names."""
            return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}
        
        @staticmethod
        def make_divisible(c: int, divisor: int) -> int:
            """Make a number divisible by the divisor."""
            return (c + divisor - 1) // divisor
        
        @staticmethod
        def calculate_zeros_width(in_features: int, group_size: int = 128, pack_num: int = 8) -> int:
            """Calculate the width of zeros for quantization."""
            base_width = AWQCore.AWQUtils.make_divisible(in_features // group_size, pack_num)
            return base_width

    class AWQLinear(nn.Module):
        """AWQ Linear layer implementation."""
        
        def __init__(
            self,
            num_bits: int,
            group_size: int,
            in_features: int,
            out_features: int,
            bias: bool,
            dev: torch.device
        ):
            """Initialize AWQ Linear layer.
            
            Args:
                num_bits: Number of bits for quantization
                group_size: Size of quantization groups
                in_features: Number of input features
                out_features: Number of output features
                bias: Whether to use bias
                dev: Device to use
            """
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.num_bits = num_bits
            self.group_size = group_size
            self.split_k_iters = 8
            self.interleave = 4
            
            pack_num = 32 // self.num_bits
            int16_pack_num = 16 // self.num_bits

            # Register buffers for quantized weights and scales
            self.register_buffer(
                "qweight",
                torch.zeros(
                    (
                        out_features // self.interleave,
                        in_features // int16_pack_num * self.interleave,
                    ),
                    dtype=torch.int16,
                    device=dev,
                ),
            )
            self.register_buffer(
                "scales",
                torch.zeros(
                    (
                        AWQCore.AWQUtils.calculate_zeros_width(in_features, self.group_size) * pack_num,
                        out_features,
                    ),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
            self.register_buffer(
                "scaled_zeros",
                torch.zeros(
                    (
                        AWQCore.AWQUtils.calculate_zeros_width(in_features, self.group_size) * pack_num,
                        out_features,
                    ),
                    dtype=torch.float16,
                    device=dev,
                ),
            )

            if bias:
                self.register_buffer(
                    "bias", torch.zeros((out_features), dtype=torch.float16, device=dev)
                )
            else:
                self.bias = None

        @classmethod
        def from_linear(
            cls,
            linear: nn.Linear,
            num_bits: int,
            group_size: int,
            scales: Optional[torch.Tensor] = None,
            zeros: Optional[torch.Tensor] = None
        ) -> 'AWQCore.AWQLinear':
            """Create AWQ Linear layer from a regular linear layer."""
            awq_linear = cls(
                num_bits,
                group_size,
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
                linear.weight.device,
            )

            scale_zeros = zeros * scales

            pack_num = 32 // awq_linear.num_bits
            qscales = torch.zeros(
                (
                    scales.shape[0],
                    AWQCore.AWQUtils.calculate_zeros_width(linear.in_features, group_size) * pack_num,
                ),
                dtype=torch.float16,
                device=scales.device,
            )
            qscales[:, : scales.shape[1]] = scales
            awq_linear.scales = qscales.transpose(1, 0).contiguous()
            if linear.bias is not None:
                awq_linear.bias = linear.bias.clone().half()

            intweight = []
            for idx in range(awq_linear.in_features):
                intweight.append(
                    torch.round(
                        (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                        / qscales[:, idx // group_size]
                    ).to(torch.int)[:, None]
                )
            intweight = torch.cat(intweight, dim=1)
            intweight = intweight.to(dtype=torch.int32)
            awq_linear.qweight = cls.pack_intweight(
                intweight.contiguous(), interleave=4, kstride=64
            )

            zeros = zeros.to(dtype=torch.int32)
            scaled_zeros = torch.zeros_like(qscales)
            scaled_zeros[:, : scales.shape[1]] = -(
                qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
            ).to(torch.float16)
            awq_linear.scaled_zeros = scaled_zeros.transpose(1, 0).contiguous()

            return awq_linear

        @staticmethod
        def pack_intweight(unpacked_qweight: torch.Tensor, interleave: int, kstride: int) -> torch.Tensor:
            """Pack integer weights for efficient storage and computation."""
            N = unpacked_qweight.shape[0]
            K = unpacked_qweight.shape[1]

            Packed_Kernel = unpacked_qweight.cpu().numpy().reshape(N, K // 32, 32)
            Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
            Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 32)

            Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 8)
            Packed_Kernel = Packed_Kernel.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
            Packed_Kernel = Packed_Kernel.reshape(N, K)

            Packed_Kernel = Packed_Kernel.reshape(
                N // interleave, interleave, K // kstride, kstride
            )
            Packed_Kernel = Packed_Kernel.transpose(0, 2, 1, 3)
            Packed_Kernel = Packed_Kernel.reshape(
                N // interleave, K // kstride, kstride, interleave
            )
            Packed_Kernel = (
                Packed_Kernel[..., 0]
                | (Packed_Kernel[..., 1] << 4)
                | (Packed_Kernel[..., 2] << 8)
                | (Packed_Kernel[..., 3] << 12)
            )
            Packed_Kernel = Packed_Kernel.reshape(N // interleave, K)
            qweight = (
                torch.tensor(Packed_Kernel.astype("int16"))
                .to(unpacked_qweight.device)
                .contiguous()
            )
            return qweight

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass of the AWQ Linear layer."""
            inputs = x
            
            if inputs.numel() / inputs.shape[-1] < 8:
                out = awq_inference_engine.gemv_forward_cuda_new(
                    inputs,
                    self.qweight,
                    self.scales,
                    self.scaled_zeros,
                    inputs.numel() // inputs.shape[-1],
                    self.out_features,
                    self.in_features,
                    self.group_size,
                )
            else:
                out = awq_inference_engine.gemm_forward_cuda_new(
                    inputs, self.qweight, self.scales, self.scaled_zeros
                )
            out = out + self.bias if self.bias is not None else out
            return out

        def extra_repr(self) -> str:
            """String representation of the layer."""
            return (
                "in_features={}, out_features={}, bias={}, num_bits={}, group_size={}".format(
                    self.in_features,
                    self.out_features,
                    self.bias is not None,
                    self.num_bits,
                    self.group_size,
                )
            )

    class AWQQuantizer:
        """Handles quantization operations for AWQ."""
        
        @staticmethod
        def quantize_tensor(
            w: torch.Tensor,
            num_bits: int,
            group_size: int,
            get_scale_zp: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            """Quantize a tensor using AWQ method.
            
            Args:
                w: Weight tensor to quantize
                num_bits: Number of bits for quantization
                group_size: Size of quantization groups
                get_scale_zp: Whether to return scales and zero points
                
            Returns:
                Quantized tensor and optionally scales and zero points
            """
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

        @staticmethod
        def quantize_model(
            model: nn.Module,
            num_bits: int,
            group_size: int
        ) -> None:
            """Quantize an entire model using AWQ method.
            
            Args:
                model: Model to quantize
                num_bits: Number of bits for quantization
                group_size: Size of quantization groups
            """
            layers = AWQCore.AWQUtils.get_blocks(model)
            for i in range(len(layers)):
                layer = layers[i]
                named_linears = AWQCore.AWQUtils.get_named_linears(layer)

                for name, module in named_linears.items():
                    module.cuda()
                    module.weight.data, scales, zeros = AWQCore.AWQQuantizer.quantize_tensor(
                        module.weight.data,
                        num_bits=num_bits,
                        group_size=group_size,
                        get_scale_zp=True
                    )
                    q_linear = AWQCore.AWQLinear.from_linear(
                        module, num_bits, group_size, scales, zeros
                    )
                    module.cpu()
                    q_linear.to(next(layer.parameters()).device)
                    AWQCore.AWQUtils.set_op_by_name(layer, name, q_linear)
                    torch.cuda.empty_cache()
                    gc.collect()

            torch.cuda.empty_cache()
            gc.collect()

    class AWQManager:
        """Main orchestration class for AWQ process.
        
        This class handles the entire AWQ workflow including:
        - Model loading and preparation
        - Finding salient weights
        - Applying scaling
        - Performing quantization
        - Evaluation
        """
        
        def __init__(
            self,
            model: nn.Module,
            tokenizer: Any,
            num_bits: int = 3,
            group_size: int = 128,
            device: str = "cuda"
        ):
            """Initialize AWQ Manager.
            
            Args:
                model: The model to quantize
                tokenizer: The tokenizer for the model
                num_bits: Number of bits for quantization
                group_size: Size of quantization groups
                device: Device to use for computation
            """
            self.model = model
            self.tokenizer = tokenizer
            self.num_bits = num_bits
            self.group_size = group_size
            self.device = device
            self.model.eval()
            
        def prepare_model(self) -> None:
            """Prepare the model for quantization."""
            self.model = self.model.to(self.device)
            AWQCore.AWQUtils.move_device(self.model, self.device)
            
        def find_salient_weights(
            self,
            calibration_data: Optional[torch.Tensor] = None,
            s_val: Optional[float] = None
        ) -> Dict[str, List]:
            """Find salient weights in the model.
            
            Args:
                calibration_data: Optional calibration data
                s_val: Optional fixed scaling value
                
            Returns:
                Dictionary containing scaling and clipping information
            """
            from find_salients import find_s_and_salient_weights
            
            return find_s_and_salient_weights(
                self.model,
                self.tokenizer,
                group_size=self.group_size,
                s_val=s_val
            )
            
        def apply_scaling(self, scaling_info: Dict[str, List]) -> None:
            """Apply scaling to the model.
            
            Args:
                scaling_info: Dictionary containing scaling information
            """
            from scale import apply_awq_scaling
            apply_awq_scaling(self.model, scaling_info)
            
        def quantize(self) -> None:
            """Quantize the model using AWQ method."""
            AWQCore.AWQQuantizer.quantize_model(
                self.model,
                num_bits=self.num_bits,
                group_size=self.group_size
            )
            
        def evaluate(self, test_data: torch.Tensor) -> float:
            """Evaluate the quantized model.
            
            Args:
                test_data: Test data for evaluation
                
            Returns:
                Perplexity score
            """
            from perplexity import compute_perplexity
            return compute_perplexity(self.model, test_data, self.device)
            
        def save_model(self, path: str) -> None:
            """Save the quantized model.
            
            Args:
                path: Path to save the model
            """
            torch.save(self.model, path)
            
        def load_model(self, path: str) -> None:
            """Load a quantized model.
            
            Args:
                path: Path to load the model from
            """
            self.model = torch.load(path)
            self.model.eval()
            
        def process(
            self,
            calibration_data: Optional[torch.Tensor] = None,
            test_data: Optional[torch.Tensor] = None,
            s_val: Optional[float] = None,
            save_path: Optional[str] = None
        ) -> Optional[float]:
            """Process the entire AWQ workflow.
            
            Args:
                calibration_data: Optional calibration data
                test_data: Optional test data for evaluation
                s_val: Optional fixed scaling value
                save_path: Optional path to save the model
                
            Returns:
                Perplexity score if test_data is provided, None otherwise
            """
            # Prepare model
            self.prepare_model()
            
            # Find salient weights
            scaling_info = self.find_salient_weights(calibration_data, s_val)
            
            # Apply scaling
            self.apply_scaling(scaling_info)
            
            # Quantize
            self.quantize()
            
            # Save if path provided
            if save_path:
                self.save_model(save_path)
                
            # Evaluate if test data provided
            if test_data is not None:
                return self.evaluate(test_data)
            return None 