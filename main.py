import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq_core import AWQCore
from awq_scale import auto_scale_block, apply_scale
from awq_clip import auto_clip_block, apply_clip
from awq_analysis import AWQAnalyzer
from awq_evaluation import AWQEvaluator
import logging
import json
from pathlib import Path
from typing import Dict, Optional

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_model_and_tokenizer(device: str = "cuda") -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load model and tokenizer.
    
    Args:
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = "facebook/opt-1.3b"  # Changed default to opt-1.3b
    logging.info(f"Loading model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def save_results(results: Dict, output_dir: str = "results") -> None:
    """Save results to file.
    
    Args:
        results: Results to save
        output_dir: Output directory
    """
    output_path = Path(output_dir) / "awq_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_path}")

def process_model(
    s_val: Optional[float] = None,
    output_dir: str = "results",
    device: str = "cuda",
    save_model: bool = True
) -> Dict:
    """Process a model using AWQ.
    
    Args:
        s_val: Optional fixed scaling value
        output_dir: Output directory
        device: Device to use
        save_model: Whether to save the quantized model
        
    Returns:
        Dictionary containing results
    """
    # Default values matching original repo
    num_bits = 3
    group_size = 128
    n_samples = 128
    seqlen = 512
    calib_data = "pileval"
    test_dataset = "wikitext2"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(device)
    
    # Create analyzer
    analyzer = AWQAnalyzer(
        model,
        tokenizer,
        num_bits=num_bits,
        group_size=group_size,
        device=device
    )
    
    # Find salient weights
    logging.info("Finding salient weights")
    s_and_salient_weights = analyzer.analyze(
        s_val=s_val,
        n_samples=n_samples,
        seqlen=seqlen,
        calib_data=calib_data
    )
    
    # Apply scaling and clipping
    logging.info("Applying scaling and clipping")
    scale_results = []
    clip_results = []
    
    for block in AWQCore.AWQUtils.get_blocks(model):
        # Apply scaling
        scale_results.append(
            auto_scale_block(
                block,
                module_kwargs={},
                num_bits=num_bits,
                group_size=group_size,
                input_feat=s_and_salient_weights,
                s_val=s_val
            )
        )
        
        # Apply clipping
        clip_results.append(
            auto_clip_block(
                block,
                num_bits=num_bits,
                group_size=group_size,
                input_feat=s_and_salient_weights
            )
        )
    
    # Combine results
    optimization_results = {
        "scale": scale_results,
        "clip": clip_results
    }
    
    # Create evaluator
    evaluator = AWQEvaluator(
        model,
        tokenizer,
        device=device,
        seqlen=seqlen
    )
    
    # Evaluate model
    logging.info("Evaluating model")
    evaluation_results = evaluator.evaluate(
        test_dataset=test_dataset,
        compute_perplexity=True,
        compute_accuracy=True,
        compute_memory=True
    )
    
    # Compare with baseline
    logging.info("Comparing with baseline")
    baseline_model, _ = load_model_and_tokenizer(device)
    comparison_results = evaluator.compare_with_baseline(
        baseline_model,
        test_dataset=test_dataset
    )
    
    # Combine results
    results = {
        "quantization": {
            "num_bits": num_bits,
            "group_size": group_size,
            "s_val": s_val
        },
        "optimization": optimization_results,
        "evaluation": evaluation_results,
        "comparison": comparison_results
    }
    
    # Save results
    save_results(results, output_dir)
    
    # Save model if requested
    if save_model:
        model_save_path = Path(output_dir) / "awq_quantized"
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    
    return results

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AWQ Model Processing")
    
    # Only keep essential arguments
    parser.add_argument(
        "--s_val",
        type=float,
        default=None,
        help="Optional fixed scaling value"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="1.3b",
        choices=["1.3b", "2.7b", "6.7b", "13b"],
        help="OPT model size to use (default: 1.3b)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Set model size
        global MODEL_NAME
        MODEL_NAME = f"facebook/opt-{args.model_size}"
        
        # Process model
        results = process_model(s_val=args.s_val)
        
        # Print summary
        logging.info("\nResults Summary:")
        logging.info(f"Model: {MODEL_NAME}")
        logging.info(f"Quantization: 3 bits, group size 128")
        logging.info(f"Perplexity: {results['evaluation']['perplexity']:.2f}")
        logging.info(f"Token Accuracy: {results['evaluation']['accuracy']['token_accuracy']:.2%}")
        logging.info(f"Memory Reduction: {results['comparison']['memory']['relative_diff']['total_size_mb']:.1f}%")
        
    except Exception as e:
        logging.error(f"Error processing model: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 