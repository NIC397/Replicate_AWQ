import torch
from torch import nn
import gc
from typing import Dict, List, Tuple, Optional, Union
from awq_core import AWQCore

class AWQOptimizer:
    """AWQ optimization class that combines scaling and clipping functionality.
    
    This class handles both the scaling and clipping operations for AWQ,
    providing a unified interface for model optimization.
    """
    
    class ScalingManager:
        """Manages the scaling operations for AWQ."""
        
        @staticmethod
        @torch.no_grad()
        def get_activation_scale(x: torch.Tensor) -> torch.Tensor:
            """Get the scale of activations.
            
            Args:
                x: Input tensor
                
            Returns:
                Scale tensor
            """
            return x.abs().view(-1, x.shape[-1]).mean(0)
        
        @staticmethod
        @torch.no_grad()
        def find_optimal_scales(
            block: nn.Module,
            linears2scale: List[nn.Linear],
            x: torch.Tensor,
            kwargs: Dict = {},
            s_val: Optional[float] = None,
            num_bits: int = 3,
            group_size: int = 128
        ) -> torch.Tensor:
            """Find optimal scaling factors for a block.
            
            Args:
                block: The block to optimize
                linears2scale: List of linear layers to scale
                x: Input tensor
                kwargs: Additional arguments for forward pass
                s_val: Optional fixed scaling value
                num_bits: Number of bits for quantization
                group_size: Size of quantization groups
                
            Returns:
                Optimal scales tensor
            """
            x = x.to(next(block.parameters()).device)
            
            # If fixed s_val, use it directly
            if s_val is not None:
                scales = torch.full((x.shape[-1],), 2, dtype=x.dtype, device=x.device)
                return scales.view(-1).detach()
            
            with torch.no_grad():
                org_out = block(x, **kwargs)
                if isinstance(org_out, tuple):
                    org_out = org_out[0]
            
            # Get average activations
            x_max = AWQOptimizer.ScalingManager.get_activation_scale(x)
            
            best_error = float("inf")
            best_scales = None
            n_grid = 20
            
            original_state = {k: v.cpu() for k, v in block.state_dict().items()}
            
            for alpha in range(n_grid):
                alpha = alpha * 1 / n_grid
                scales = x_max.pow(alpha).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                
                for fc in linears2scale:
                    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                    fc.weight.data = AWQCore.AWQQuantizer.quantize_tensor(
                        fc.weight.data,
                        num_bits=num_bits,
                        group_size=group_size
                    ) / (scales.view(1, -1))
                
                out = block(x, **kwargs)
                if isinstance(out, tuple):
                    out = out[0]
                
                loss = (org_out - out).float().pow(2).mean().item()
                
                if loss < best_error:
                    best_error = loss
                    best_scales = scales
                
                block.load_state_dict(original_state)
            
            return best_scales.view(-1).detach()
        
        @staticmethod
        @torch.no_grad()
        def apply_scaling(
            module: nn.Module,
            scales_list: List[Tuple],
            input_feat_dict: Optional[Dict] = None
        ) -> None:
            """Apply scaling to the module.
            
            Args:
                module: Module to scale
                scales_list: List of scaling information
                input_feat_dict: Optional input feature dictionary
            """
            for prev_op_name, layer_names, scales in scales_list:
                prev_op = AWQCore.AWQUtils.get_op_by_name(module, prev_op_name)
                layers = [AWQCore.AWQUtils.get_op_by_name(module, name) for name in layer_names]
                
                prev_op.cuda()
                for layer in layers:
                    layer.cuda()
                scales.cuda()
                
                if isinstance(prev_op, nn.Linear):
                    AWQOptimizer.ScalingManager._scale_fc_fc(prev_op, layers[0], scales)
                elif isinstance(prev_op, nn.LayerNorm):
                    AWQOptimizer.ScalingManager._scale_ln_fcs(prev_op, layers, scales)
                
                if input_feat_dict is not None:
                    for layer_name in layer_names:
                        inp = input_feat_dict[layer_name]
                        inp.div_(scales.view(1, -1).to(inp.device))
        
        @staticmethod
        @torch.no_grad()
        def _scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor) -> None:
            """Scale between two fully connected layers."""
            scales = scales.to(fc1.weight.device)
            fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
            if fc1.bias is not None:
                fc1.bias.div_(scales.view(-1))
            fc2.weight.mul_(scales.view(1, -1))
        
        @staticmethod
        @torch.no_grad()
        def _scale_ln_fcs(ln: nn.LayerNorm, fcs: Union[nn.Linear, List[nn.Linear]], scales: torch.Tensor) -> None:
            """Scale between layer norm and fully connected layers."""
            if not isinstance(fcs, list):
                fcs = [fcs]
            
            scales = scales.to(ln.weight.device)
            ln.weight.div_(scales)
            if hasattr(ln, "bias") and ln.bias is not None:
                ln.bias.div_(scales)
            
            for fc in fcs:
                fc.weight.mul_(scales.view(1, -1))
    
    class ClippingManager:
        """Manages the clipping operations for AWQ."""
        
        @staticmethod
        @torch.no_grad()
        def find_optimal_clip(
            w: torch.Tensor,
            input_feat: torch.Tensor,
            num_bits: int,
            group_size: int,
            n_grid: int = 20,
            max_shrink: float = 0.5,
            n_sample_token: int = 512
        ) -> torch.Tensor:
            """Find optimal clipping values for weights.
            
            Args:
                w: Weight tensor
                input_feat: Input features
                num_bits: Number of bits for quantization
                group_size: Size of quantization groups
                n_grid: Number of grid points for search
                max_shrink: Maximum shrinkage factor
                n_sample_token: Number of tokens to sample
                
            Returns:
                Optimal clipping values
            """
            input_feat = input_feat.view(-1, input_feat.shape[-1])
            input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
            input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
            w = w.reshape(w.shape[0], 1, -1, group_size)
            
            oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
            w_all = w
            best_max_val_all = []
            
            for i_b in range(w.shape[0] // oc_batch_size):
                w = w_all[i_b * oc_batch_size:(i_b + 1) * oc_batch_size]
                
                org_max_val = w.abs().amax(dim=-1, keepdim=True)
                best_max_val = org_max_val.clone()
                min_errs = torch.ones_like(org_max_val) * 1e9
                input_feat = input_feat.to(w.device)
                org_out = (input_feat * w).sum(dim=-1)
                
                for i_s in range(int(max_shrink * n_grid)):
                    max_val = org_max_val * (1 - i_s / n_grid)
                    min_val = -max_val
                    cur_w = torch.clamp(w, min_val, max_val)
                    q_w = AWQCore.AWQQuantizer.quantize_tensor(
                        cur_w,
                        num_bits=num_bits,
                        group_size=group_size
                    )
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
        
        @staticmethod
        @torch.no_grad()
        def apply_clipping(module: nn.Module, clip_list: List[Tuple[str, torch.Tensor]]) -> None:
            """Apply clipping to the module.
            
            Args:
                module: Module to clip
                clip_list: List of clipping information
            """
            for name, max_val in clip_list:
                layer = AWQCore.AWQUtils.get_op_by_name(module, name)
                layer.cuda()
                max_val = max_val.to(layer.weight.device)
                org_shape = layer.weight.shape
                layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
                layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
                layer.weight.data = layer.weight.data.reshape(org_shape)
                layer.cpu()
            
            gc.collect()
            torch.cuda.empty_cache()
    
    def __init__(
        self,
        model: nn.Module,
        num_bits: int = 3,
        group_size: int = 128,
        device: str = "cuda"
    ):
        """Initialize AWQ Optimizer.
        
        Args:
            model: Model to optimize
            num_bits: Number of bits for quantization
            group_size: Size of quantization groups
            device: Device to use
        """
        self.model = model
        self.num_bits = num_bits
        self.group_size = group_size
        self.device = device
        self.scaling_manager = AWQOptimizer.ScalingManager()
        self.clipping_manager = AWQOptimizer.ClippingManager()
    
    def optimize(
        self,
        input_feat: Dict[str, torch.Tensor],
        module_kwargs: Dict = {},
        s_val: Optional[float] = None
    ) -> Dict[str, List]:
        """Optimize the model using both scaling and clipping.
        
        Args:
            input_feat: Input features dictionary
            module_kwargs: Additional arguments for forward pass
            s_val: Optional fixed scaling value
            
        Returns:
            Dictionary containing optimization results
        """
        # Get scaling information
        scales_list = self._get_scaling_info(input_feat, module_kwargs, s_val)
        
        # Get clipping information
        clip_list = self._get_clipping_info(input_feat)
        
        # Apply optimizations
        self.scaling_manager.apply_scaling(self.model, scales_list, input_feat)
        self.clipping_manager.apply_clipping(self.model, clip_list)
        
        return {
            "scale": scales_list,
            "clip": clip_list
        }
    
    def _get_scaling_info(
        self,
        input_feat: Dict[str, torch.Tensor],
        module_kwargs: Dict,
        s_val: Optional[float]
    ) -> List[Tuple]:
        """Get scaling information for the model."""
        scales_list = []
        
        # Process self-attention layers
        scales_list.append(
            self._get_layer_scales(
                prev_op=self.model.self_attn_layer_norm,
                layers=[
                    self.model.self_attn.q_proj,
                    self.model.self_attn.k_proj,
                    self.model.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                inspect_module=self.model.self_attn,
                kwargs=module_kwargs,
                s_val=s_val
            )
        )
        
        # Process output projection
        scales_list.append(
            self._get_layer_scales(
                prev_op=self.model.self_attn.v_proj,
                layers=[self.model.self_attn.out_proj],
                inp=input_feat["self_attn.out_proj"],
                inspect_module=self.model.self_attn.out_proj,
                s_val=s_val
            )
        )
        
        # Process feed-forward layers
        scales_list.append(
            self._get_layer_scales(
                prev_op=self.model.final_layer_norm,
                layers=[self.model.fc1],
                inp=input_feat["fc1"],
                inspect_module=self.model.fc1,
                s_val=s_val
            )
        )
        
        scales_list.append(
            self._get_layer_scales(
                prev_op=self.model.fc1,
                layers=[self.model.fc2],
                inp=input_feat["fc2"],
                inspect_module=self.model.fc2,
                s_val=s_val
            )
        )
        
        return scales_list
    
    def _get_layer_scales(
        self,
        prev_op: nn.Module,
        layers: List[nn.Module],
        inp: torch.Tensor,
        inspect_module: nn.Module,
        kwargs: Dict = {},
        s_val: Optional[float] = None
    ) -> Tuple:
        """Get scaling information for a specific layer."""
        scales = self.scaling_manager.find_optimal_scales(
            inspect_module,
            layers,
            inp,
            kwargs,
            s_val,
            self.num_bits,
            self.group_size
        )
        scales = scales.detach().cpu()
        
        return (
            AWQCore.AWQUtils.get_op_name(self.model, prev_op),
            tuple([AWQCore.AWQUtils.get_op_name(self.model, m) for m in layers]),
            scales,
        )
    
    def _get_clipping_info(self, input_feat: Dict[str, torch.Tensor]) -> List[Tuple[str, torch.Tensor]]:
        """Get clipping information for the model."""
        named_linears = AWQCore.AWQUtils.get_named_linears(self.model)
        clip_list = []
        
        for name, module in named_linears.items():
            # Skip attention layers
            if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
                continue
            
            module.cuda()
            max_val = self.clipping_manager.find_optimal_clip(
                module.weight,
                input_feat[name],
                self.num_bits,
                self.group_size
            )
            clip_list.append((name, max_val))
            module.cpu()
        
        return clip_list 