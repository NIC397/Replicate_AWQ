import torch
from torch import nn
from tqdm import tqdm

def compute_perplexity(language_model, evaluation_data, compute_device):
    language_model.seqlen = 2048
    evaluation_data = evaluation_data.input_ids.to(compute_device)
    total_chunks = evaluation_data.numel() // language_model.seqlen
    language_model = language_model.eval()
    
    negative_log_likelihoods = []
    
    for chunk_idx in tqdm(range(total_chunks), desc="Computing perplexity..."):
        input_sequence = evaluation_data[:, (chunk_idx * language_model.seqlen):((chunk_idx + 1) * language_model.seqlen)].to(
            compute_device
        )
        
        with torch.no_grad():
            output_logits = language_model(input_sequence.cuda()).logits
            
        prediction_logits = output_logits[:, :-1, :].contiguous().float()
        target_tokens = evaluation_data[
            :, (chunk_idx * language_model.seqlen):((chunk_idx + 1) * language_model.seqlen)
        ][:, 1:]
        
        cross_entropy = nn.CrossEntropyLoss()
        sequence_loss = cross_entropy(
            prediction_logits.reshape(-1, prediction_logits.size(-1)), 
            target_tokens.reshape(-1)
        )
        
        chunk_nll = sequence_loss.float() * language_model.seqlen
        negative_log_likelihoods.append(chunk_nll)

    perplexity_score = torch.exp(torch.stack(negative_log_likelihoods).sum() / (total_chunks * language_model.seqlen))
    return perplexity_score