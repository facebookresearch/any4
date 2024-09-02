from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math
import torch
from tqdm import tqdm

task_dataset_configs = {
    "wikitext-2":   {"dataset": "wikitext", "config": "wikitext-2-raw-v1", "split": "test", "field": "text"},
    "c4":           {"dataset": "allenai/c4", "config": "en", "split": "validation", "field": "text"},
    "pile":         {"dataset": "EleutherAI/pile", "config": "all", "split": "test", "field": "text"},
    "ptb":          {"dataset": "ptb-text-only/ptb_text_only", "config": "all", "split": "test", "field": "text"},
}

def eval_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Optional[str] = "wikitext",
    config: Optional[str] = "wikitext-2-raw-v1",
    split: str = "test",
    field: str = "text",
    seed: int = 42,
    batch_size: int = 8,
    num_batches: Optional[int] = None,
    max_seq_len: Optional[int] = 2048,
):
    model.eval()  # Set the model to evaluation mode

    # Load dataset in streaming mode
    dataset = load_dataset(dataset, config, split=split, streaming=True).shuffle(seed=seed)

    def calculate_perplexity(model, encoded_chunks):
        total_loss = 0
        total_length = 0

        # TODO: check in code
        # TODO: pass as single batch rather than slicing
        # TODO: debug nan loss
        with torch.no_grad():
            for chunk in encoded_chunks:
                inputs = torch.tensor(chunk["input_ids"]).to(model.device).unsqueeze(0)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss.item()
                total_loss += loss * inputs.shape[1]
                total_length += inputs.shape[1]

        average_loss = total_loss / total_length
        perplexity = math.exp(average_loss)
        return perplexity

    perplexities = []
    encoded_chunks = []

    for example in tqdm(dataset):
        # Tokenize the text
        tokenized_example = tokenizer(example[field], return_special_tokens_mask=True)
        
        # Handle sequences longer than max_seq_len
        input_ids = tokenized_example["input_ids"]
        for i in range(0, len(input_ids), max_seq_len):
            chunk = input_ids[i:i + max_seq_len]
            encoded_chunks.append({"input_ids": chunk})
            
            # Check if we have enough chunks to form a batch
            if len(encoded_chunks) >= batch_size:
                # Process exactly batch_size chunks
                batch_chunks = encoded_chunks[:batch_size]
                perplexity = calculate_perplexity(model, batch_chunks)
                perplexities.append(perplexity)
                
                # Remove the processed chunks
                encoded_chunks = encoded_chunks[batch_size:]

            if num_batches is not None:
                if len(perplexities) >= num_batches * batch_size:
                    break

    # Handle any remaining chunks in the last batch
    if encoded_chunks:
        perplexity = calculate_perplexity(model, encoded_chunks)
        perplexities.append(perplexity)

    # Overall average perplexity
    overall_perplexity = sum(perplexities) / len(perplexities)
    print(f"Overall Perplexity: {overall_perplexity}")
    return overall_perplexity
