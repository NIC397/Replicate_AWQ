import torch
from torch import nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from datasets import load_dataset
import numpy as np
from awq_core import AWQCore

class AWQEvaluator:
    """AWQ evaluation class that provides comprehensive model evaluation.
    
    This class handles model evaluation including perplexity calculation,
    accuracy metrics, and memory usage analysis.
    """
    
    class PerplexityEvaluator:
        """Manages perplexity evaluation for AWQ models."""
        
        @staticmethod
        @torch.no_grad()
        def compute_perplexity(
            model: nn.Module,
            test_encodings,
            device: str = "cuda",
            seqlen: int = 2048
        ) -> float:
            """Compute perplexity score for the model.
            
            Args:
                model: Model to evaluate
                test_encodings: Test dataset encodings
                device: Device to use
                seqlen: Sequence length
                
            Returns:
                Perplexity score
            """
            model.seqlen = seqlen
            test_encodings = test_encodings.input_ids.to(device)
            nsamples = test_encodings.numel() // model.seqlen
            model = model.eval()
            nlls = []
            
            for i in tqdm(range(nsamples), desc="Evaluating perplexity..."):
                batch = test_encodings[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(device)
                
                with torch.no_grad():
                    lm_logits = model(batch).logits
                
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = test_encodings[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)
            
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            return ppl.item()
    
    class AccuracyEvaluator:
        """Manages accuracy evaluation for AWQ models."""
        
        @staticmethod
        @torch.no_grad()
        def compute_accuracy(
            model: nn.Module,
            test_encodings,
            device: str = "cuda",
            seqlen: int = 2048
        ) -> Dict[str, float]:
            """Compute accuracy metrics for the model.
            
            Args:
                model: Model to evaluate
                test_encodings: Test dataset encodings
                device: Device to use
                seqlen: Sequence length
                
            Returns:
                Dictionary containing accuracy metrics
            """
            model.seqlen = seqlen
            test_encodings = test_encodings.input_ids.to(device)
            nsamples = test_encodings.numel() // model.seqlen
            model = model.eval()
            
            total_tokens = 0
            correct_tokens = 0
            total_sequences = 0
            correct_sequences = 0
            
            for i in tqdm(range(nsamples), desc="Evaluating accuracy..."):
                batch = test_encodings[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(device)
                
                with torch.no_grad():
                    lm_logits = model(batch).logits
                
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = test_encodings[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                
                # Token-level accuracy
                predictions = shift_logits.argmax(dim=-1)
                correct_tokens += (predictions == shift_labels).sum().item()
                total_tokens += shift_labels.numel()
                
                # Sequence-level accuracy
                sequence_correct = (predictions == shift_labels).all(dim=1)
                correct_sequences += sequence_correct.sum().item()
                total_sequences += sequence_correct.size(0)
            
            return {
                "token_accuracy": correct_tokens / total_tokens,
                "sequence_accuracy": correct_sequences / total_sequences
            }
    
    class MemoryEvaluator:
        """Manages memory usage evaluation for AWQ models."""
        
        @staticmethod
        def compute_memory_usage(
            model: nn.Module,
            device: str = "cuda"
        ) -> Dict[str, float]:
            """Compute memory usage statistics for the model.
            
            Args:
                model: Model to evaluate
                device: Device to use
                
            Returns:
                Dictionary containing memory usage statistics
            """
            # Move model to device
            model = model.to(device)
            
            # Get model size
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            # Get peak memory usage
            torch.cuda.reset_peak_memory_stats()
            _ = model(torch.zeros(1, 1, dtype=torch.long, device=device))
            peak_memory = torch.cuda.max_memory_allocated()
            
            return {
                "parameter_size_mb": param_size / (1024 * 1024),
                "buffer_size_mb": buffer_size / (1024 * 1024),
                "total_size_mb": (param_size + buffer_size) / (1024 * 1024),
                "peak_memory_mb": peak_memory / (1024 * 1024)
            }
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        seqlen: int = 2048
    ):
        """Initialize AWQ Evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer to use
            device: Device to use
            seqlen: Sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seqlen = seqlen
        self.perplexity_evaluator = AWQEvaluator.PerplexityEvaluator()
        self.accuracy_evaluator = AWQEvaluator.AccuracyEvaluator()
        self.memory_evaluator = AWQEvaluator.MemoryEvaluator()
    
    def evaluate(
        self,
        test_dataset: str = "wikitext2",
        compute_perplexity: bool = True,
        compute_accuracy: bool = True,
        compute_memory: bool = True
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Evaluate the model using specified metrics.
        
        Args:
            test_dataset: Test dataset to use
            compute_perplexity: Whether to compute perplexity
            compute_accuracy: Whether to compute accuracy
            compute_memory: Whether to compute memory usage
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {}
        
        # Load test dataset
        if test_dataset == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        else:
            raise NotImplementedError(f"Dataset {test_dataset} not implemented")
        
        # Prepare test encodings
        test_encodings = self.tokenizer(
            "\n\n".join(dataset["text"]),
            return_tensors="pt"
        )
        
        # Compute metrics
        if compute_perplexity:
            results["perplexity"] = self.perplexity_evaluator.compute_perplexity(
                self.model,
                test_encodings,
                self.device,
                self.seqlen
            )
        
        if compute_accuracy:
            results["accuracy"] = self.accuracy_evaluator.compute_accuracy(
                self.model,
                test_encodings,
                self.device,
                self.seqlen
            )
        
        if compute_memory:
            results["memory"] = self.memory_evaluator.compute_memory_usage(
                self.model,
                self.device
            )
        
        return results
    
    def compare_with_baseline(
        self,
        baseline_model: nn.Module,
        test_dataset: str = "wikitext2"
    ) -> Dict[str, Dict[str, float]]:
        """Compare model performance with a baseline model.
        
        Args:
            baseline_model: Baseline model to compare with
            test_dataset: Test dataset to use
            
        Returns:
            Dictionary containing comparison results
        """
        # Evaluate both models
        quantized_results = self.evaluate(test_dataset)
        
        baseline_evaluator = AWQEvaluator(
            baseline_model,
            self.tokenizer,
            self.device,
            self.seqlen
        )
        baseline_results = baseline_evaluator.evaluate(test_dataset)
        
        # Compute relative differences
        comparison = {}
        
        if "perplexity" in quantized_results:
            comparison["perplexity"] = {
                "quantized": quantized_results["perplexity"],
                "baseline": baseline_results["perplexity"],
                "relative_diff": (
                    quantized_results["perplexity"] - baseline_results["perplexity"]
                ) / baseline_results["perplexity"] * 100
            }
        
        if "accuracy" in quantized_results:
            comparison["accuracy"] = {
                "quantized": quantized_results["accuracy"],
                "baseline": baseline_results["accuracy"],
                "relative_diff": {
                    metric: (
                        quantized_results["accuracy"][metric] -
                        baseline_results["accuracy"][metric]
                    ) / baseline_results["accuracy"][metric] * 100
                    for metric in quantized_results["accuracy"]
                }
            }
        
        if "memory" in quantized_results:
            comparison["memory"] = {
                "quantized": quantized_results["memory"],
                "baseline": baseline_results["memory"],
                "relative_diff": {
                    metric: (
                        quantized_results["memory"][metric] -
                        baseline_results["memory"][metric]
                    ) / baseline_results["memory"][metric] * 100
                    for metric in quantized_results["memory"]
                }
            }
        
        return comparison 