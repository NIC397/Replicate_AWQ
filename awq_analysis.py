import torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict
import functools
import gc
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional, Union
from awq_core import AWQCore
from awq_optimization import AWQOptimizer

class AWQAnalyzer:
    """AWQ analysis class that combines calibration and salient weight finding functionality.
    
    This class handles both the calibration data preparation and finding salient weights
    for AWQ, providing a unified interface for model analysis.
    """
    
    class CalibrationManager:
        """Manages the calibration data preparation for AWQ."""
        
        @staticmethod
        def get_calibration_dataset(
            data_source: str,
            tokenizer,
            n_samples: int = 128,
            block_size: int = 512
        ) -> List[torch.Tensor]:
            """Get calibration dataset for AWQ.
            
            Args:
                data_source: Source of calibration data
                tokenizer: Tokenizer to use
                n_samples: Number of samples to use
                block_size: Size of each block
                
            Returns:
                List of calibration data blocks
            """
            if data_source == "pileval":
                dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
            else:
                raise NotImplementedError(f"Data source {data_source} not implemented")
            
            dataset = dataset.shuffle(seed=42)
            samples = []
            n_run = 0
            
            for data in dataset:
                line = data["text"].strip()
                line_encoded = tokenizer.encode(line)
                
                if len(line_encoded) > block_size:
                    continue
                    
                sample = torch.tensor([line_encoded])
                if sample.numel() == 0:
                    continue
                    
                samples.append(sample)
                n_run += 1
                if n_run == n_samples:
                    break
            
            # Concatenate and split samples
            cat_samples = torch.cat(samples, dim=1)
            n_split = cat_samples.shape[1] // block_size
            
            return [
                cat_samples[:, i * block_size : (i + 1) * block_size]
                for i in range(n_split)
            ]
    
    class SalientWeightFinder:
        """Manages finding salient weights for AWQ."""
        
        @staticmethod
        @torch.no_grad()
        def find_salient_weights(
            model: nn.Module,
            tokenizer,
            group_size: int,
            s_val: Optional[float] = None,
            num_bits: int = 3,
            n_samples: int = 128,
            seqlen: int = 512,
            calib_data: str = "pileval"
        ) -> Dict[str, List]:
            """Find salient weights and scaling factors for the model.
            
            Args:
                model: Model to analyze
                tokenizer: Tokenizer to use
                group_size: Size of quantization groups
                s_val: Optional fixed scaling value
                num_bits: Number of bits for quantization
                n_samples: Number of calibration samples
                seqlen: Sequence length
                calib_data: Calibration data source
                
            Returns:
                Dictionary containing scaling and clipping information
            """
            # Get model blocks
            layers = AWQCore.AWQUtils.get_blocks(model)
            
            # Get calibration data
            samples = AWQAnalyzer.CalibrationManager.get_calibration_dataset(
                calib_data,
                tokenizer,
                n_samples,
                seqlen
            )
            samples = torch.cat(samples, dim=0)
            
            # Prepare for input capture
            inps = []
            layer_kwargs = {}
            
            # Move first layer to GPU
            layers[0] = layers[0].cuda()
            AWQCore.AWQUtils.move_device(model, "cuda")
            
            # Input capture setup
            class InputCatcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                
                def forward(self, inp, **kwargs):
                    inps.append(inp)
                    layer_kwargs.update(kwargs)
                    raise ValueError
            
            layers[0] = InputCatcher(layers[0])
            try:
                model(samples.to(next(model.parameters()).device))
            except ValueError:
                pass
            
            del samples
            layers[0] = layers[0].module
            inps = inps[0]
            
            # Move back to CPU
            layers[0] = layers[0].cpu()
            AWQCore.AWQUtils.move_device(model, "cpu")
            
            gc.collect()
            
            # Initialize results
            s_and_salient_weights = {
                "scale": [],
                "clip": [],
            }
            torch.cuda.empty_cache()
            
            # Process each layer
            for i in tqdm(range(len(layers)), desc="Running AWQ..."):
                # Clear GPU Memory
                gc.collect()
                torch.cuda.empty_cache()
                
                layer = layers[i]
                layer = layer.cuda()
                named_linears = AWQCore.AWQUtils.get_named_linears(layer)
                
                # Cache input features
                input_feat = defaultdict(list)
                handles = []
                
                def cache_input_hook(m, x, y, name, feat_dict):
                    x = x[0]
                    x = x.detach().cpu()
                    feat_dict[name].append(x)
                
                for name in named_linears:
                    handles.append(
                        named_linears[name].register_forward_hook(
                            functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                        )
                    )
                
                inps = layer(inps, **layer_kwargs)[0]
                
                for h in handles:
                    h.remove()
                del handles
                
                input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
                
                # Clear GPU memory
                gc.collect()
                torch.cuda.empty_cache()
                
                # Create optimizer for this layer
                optimizer = AWQOptimizer(
                    layer,
                    num_bits=num_bits,
                    group_size=group_size
                )
                
                # Get optimization results
                results = optimizer.optimize(
                    input_feat=input_feat,
                    module_kwargs=layer_kwargs,
                    s_val=s_val
                )
                
                # Update results
                s_and_salient_weights["scale"] += AWQCore.AWQUtils.append_str_prefix(
                    results["scale"],
                    AWQCore.AWQUtils.get_op_name(model, layer) + "."
                )
                
                s_and_salient_weights["clip"] += AWQCore.AWQUtils.append_str_prefix(
                    results["clip"],
                    AWQCore.AWQUtils.get_op_name(model, layer) + "."
                )
                
                # Cleanup
                layer = layer.cpu()
                del layer
                del input_feat
                del results
                del named_linears
                gc.collect()
                torch.cuda.empty_cache()
            
            return s_and_salient_weights
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_bits: int = 3,
        group_size: int = 128,
        device: str = "cuda"
    ):
        """Initialize AWQ Analyzer.
        
        Args:
            model: Model to analyze
            tokenizer: Tokenizer to use
            num_bits: Number of bits for quantization
            group_size: Size of quantization groups
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_bits = num_bits
        self.group_size = group_size
        self.device = device
        self.calibration_manager = AWQAnalyzer.CalibrationManager()
        self.salient_finder = AWQAnalyzer.SalientWeightFinder()
    
    def analyze(
        self,
        s_val: Optional[float] = None,
        n_samples: int = 128,
        seqlen: int = 512,
        calib_data: str = "pileval"
    ) -> Dict[str, List]:
        """Analyze the model to find salient weights and scaling factors.
        
        Args:
            s_val: Optional fixed scaling value
            n_samples: Number of calibration samples
            seqlen: Sequence length
            calib_data: Calibration data source
            
        Returns:
            Dictionary containing scaling and clipping information
        """
        return self.salient_finder.find_salient_weights(
            self.model,
            self.tokenizer,
            self.group_size,
            s_val,
            self.num_bits,
            n_samples,
            seqlen,
            calib_data
        ) 