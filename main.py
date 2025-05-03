from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from datasets import load_dataset
from awq_analysis import find_s_and_salient_weights
from awq_utils import quantize
from awq_evaluation import compute_perplexity
from awq_optimization import apply_awq_scaling
import gc
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--s_val', type=int, default=None, help='Pass in a fixed s_val, avoid the search. Default behavior is searching for s.')
parser.add_argument('--test', action='store_true', help='Run the script in test mode.')

args = parser.parse_args()

if args.test:
    models = ["opt-125m"]
else:
    models = ["opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b"]


group_size = 128
kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
num_bits = 3
s_val = args.s_val

if __name__ == "__main__":
    testset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    perplexities = {}

    for model_name in models:
        gc.collect()
        torch.cuda.empty_cache()
        model_path = "facebook/" + model_name
        print("Performing AWQ on " + model_name + "...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        enc = AutoTokenizer.from_pretrained(
                        model_path, use_fast=False, trust_remote_code=True
                    )
        
        model = AutoModelForCausalLM.from_pretrained(
                        model_path, config=config, trust_remote_code=True, **kwargs
                    )
        model.eval()

        s_and_salient_weights = find_s_and_salient_weights(model,
                        enc,
                        group_size=group_size,
                        s_val=s_val)

        # Reset model
        model = AutoModelForCausalLM.from_pretrained(
                        model_path, config=config, trust_remote_code=True, **kwargs
                    )
        model.eval()

        apply_awq_scaling(model, s_and_salient_weights)
        quantize(model, num_bits=num_bits, group_size=group_size)
        if args.s_val is not None:
            torch.save(model, model_name + '_s2.pt')
        else:
            torch.save(model, model_name + "_awq.pt")

        testenc = enc("\n\n".join(testset["text"]), return_tensors="pt")

        model.to('cuda')
        perplexity = compute_perplexity(model, testenc, 'cuda')
        perplexities[model_name] = perplexity.item()
        print()
        print(perplexity.item())

    print("Summary of AWQ Perplexities")
    for k,v in perplexities.items():
        print(f"Perplexity for {k} with AWQ to {num_bits} bits: {v}")
