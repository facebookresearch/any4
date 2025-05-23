# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm

task_dataset_configs = {
    "wikitext-2":           {"dataset": "wikitext", "config": "wikitext-2-raw-v1", "split": "test", "field": "text", "separator": "\n\n"},
    "wikipedia":            {"dataset": "wikimedia/wikipedia", "config": "20231101.en", "split": "train", "field": "text"},
    "c4":                   {"dataset": "allenai/c4", "config": "en", "split": "validation", "field": "text"},
    "pile-clean":           {"dataset": "monology/pile-uncopyrighted", "split": "train", "field": "text"},
    "ptb":                  {"dataset": "ptb-text-only/ptb_text_only", "config": "penn_treebank", "split": "test", "field": "sentence", "separator": "\n\n"},
    "codeparrot":           {"dataset": "huggingface-course/codeparrot-ds-valid", "split": "validation", "field": "content"},
    "pile-arxiv":           {"dataset": "ArmelR/the-pile-splitted", "config": "ArXiv", "split": "test", "field": "text"},
    "pile-books":           {"dataset": "ArmelR/the-pile-splitted", "config": "BookCorpus2", "split": "test", "field": "text"},
    "pile-books3":          {"dataset": "ArmelR/the-pile-splitted", "config": "Books3", "split": "test", "field": "text"},
    "pile-math":            {"dataset": "ArmelR/the-pile-splitted", "config": "DM Mathematics", "split": "test", "field": "text"},
    "pile-enron":           {"dataset": "ArmelR/the-pile-splitted", "config": "Enron Emails", "split": "test", "field": "text"},
    "pile-europarl":        {"dataset": "ArmelR/the-pile-splitted", "config": "EuroParl", "split": "test", "field": "text"},
    "pile-freelaw":         {"dataset": "ArmelR/the-pile-splitted", "config": "FreeLaw", "split": "test", "field": "text"},
    "pile-github":          {"dataset": "ArmelR/the-pile-splitted", "config": "GitHub", "split": "test", "field": "text"},
    "pile-subtitles":       {"dataset": "ArmelR/the-pile-splitted", "config": "OpenSubtitles", "split": "test", "field": "text"},
    "pile-openwebtext2":    {"dataset": "ArmelR/the-pile-splitted", "config": "OpenWebText2", "split": "test", "field": "text"},
    "pile-pubmed-central":  {"dataset": "ArmelR/the-pile-splitted", "config": "PubMed Central", "split": "test", "field": "text"},
    "pile-stackexchange":   {"dataset": "ArmelR/the-pile-splitted", "config": "StackExchange", "split": "test", "field": "text"},
    "pile-uspto":           {"dataset": "ArmelR/the-pile-splitted", "config": "USPTO Backgrounds", "split": "test", "field": "text"},
    "pile-ubuntu-irc":      {"dataset": "ArmelR/the-pile-splitted", "config": "Ubuntu IRC", "split": "test", "field": "text"},
    "pile-youtube":         {"dataset": "ArmelR/the-pile-splitted", "config": "YoutubeSubtitles", "split": "test", "field": "text"},
}

def eval_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: Optional[str] = "wikitext",
    config: Optional[str] = None,
    split: str = "test",
    field: str = "text",
    seed: int = None,
    batch_size: int = 8,
    num_batches: Optional[int] = 256,
    max_seq_len: Optional[int] = 2048,
    separator: Optional[str] = "",
    handle_tail_samples: Optional[bool] = False,
):
    print(f"Evaluating perplexity on {dataset} on split {split}...")
    model.eval()  # Set the model to evaluation mode

    # Load dataset
    if separator:
        dataset = load_dataset(dataset, config, split=split)
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        dataset = Dataset.from_dict({field: ["\n\n".join(dataset[field])]})
    else:
        dataset = load_dataset(dataset, config, split=split, streaming=True)
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

    long_sequence = []
    batch_count = 0
    losses = []
    total_seq_len = 0

    with tqdm(dataset) as pbar:
        for example in pbar:
            # Tokenize the text
            tokenized_example = tokenizer(example[field], return_special_tokens_mask=True)

            # Append tokenized input IDs to the long sequence
            # TODO: check if <BOS> and <EOS> are inserted accordingly between concatenated samples.
            long_sequence.extend(tokenized_example["input_ids"])

            # Check if the long sequence exceeds the maximum sequence length times the number of batches
            # if len(long_sequence) >= max_seq_len * batch_size:
            for idx in range(0, len(long_sequence), max_seq_len * batch_size):
                if idx + max_seq_len * batch_size > len(long_sequence):
                    break
                # Reshape the long sequence into a tensor of size <batch_size, max_seq_len>
                input_ids = long_sequence[idx : idx + max_seq_len * batch_size]
                input_ids = torch.tensor(input_ids, device=model.device).view(batch_size, max_seq_len)

                # Calculate perplexity
                with torch.no_grad():
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    losses.append(loss * input_ids.numel())
                    total_seq_len += input_ids.numel()
                pbar.set_postfix({"loss": loss.item()})

                # Update batch count and check if it exceeds num_batches
                batch_count += 1
                if num_batches is not None and batch_count >= num_batches:
                    break

            # Remove the processed sequences
            long_sequence = long_sequence[idx:]

            # Check if it exceeds num_batches
            if num_batches is not None and batch_count >= num_batches:
                break

    # Handle any remaining sequence if batch count hasn't reached num_batches
    if handle_tail_samples:
        if long_sequence and num_batches is not None and batch_count < num_batches:
            # Pad the remaining sequence to max_seq_len
            padding_length = max_seq_len - (len(long_sequence) % max_seq_len)
            long_sequence.extend([tokenizer.pad_token_id] * padding_length)

            # Reshape the remaining sequence into a tensor of size <1, max_seq_len>
            input_ids = torch.tensor(long_sequence, device=model.device).view(-1, max_seq_len)

            # Calculate perplexity
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                losses.append(loss * input_ids.numel())
                total_seq_len += input_ids.numel()

    eval_loss = torch.sum(torch.Tensor(losses)) / total_seq_len
    perplexity = torch.exp(eval_loss).item()

    print(f"Perplexity: {perplexity}")

    return perplexity
