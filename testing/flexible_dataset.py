import datasets
import torch
import random

from datasets import load_dataset
from torch.utils.data import Dataset


class FlexibleDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, model, seq_len=120, num_samples=64, random_seed=32):
        datasets.enable_caching()

        self.tokenizer = tokenizer
        self.max_length = seq_len
        self.dataset_option = dataset_name
        torch.manual_seed(random_seed)

        if self.dataset_option == "bookcorpus":
            self.dataset = load_dataset("bookcorpus/bookcorpus", split=f"train[:{num_samples}]", streaming=False, trust_remote_code=True, cache_dir="/cache")
        elif self.dataset_option == "wikitext":
            self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{num_samples}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "sst2":
            self.dataset = load_dataset("glue", "sst2", split=f"train[:{num_samples}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "wmt19":
            self.dataset = load_dataset("wmt/wmt19", "de-en", split=f"train[:{num_samples}]", streaming=False, cache_dir="/cache")
        elif self.dataset_option == "arxiver":
            self.dataset = load_dataset("neuralwork/arxiver", split=f"train[:{num_samples}]", straming=False, cache_dir="/cache")
        elif self.dataset_option == "random":
            pass
        else:
            raise ValueError("Invalid dataset option")

        if self.dataset_option != "random":
            self.dataset_size = len(self.dataset)
        else:
            self.dataset_size = num_samples

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.dataset_option == "bookcorpus" or self.dataset_option == "wikitext":
            encoder = "summarize: " + self.dataset[idx]["text"]
        elif self.dataset_option == "sst2":
            encoder = "summarize: " + self.dataset[idx]["sentence"]
        elif self.dataset_option == "arxiver":
            encoder = "summarize: " + self.dataset[idx]["abstract"]
        elif self.dataset_option == "wmt19":
            encoder = "translate English to German: " + self.dataset[idx]["translation"]["en"]
        elif self.dataset_option == "random":
            encoder = ["summarize:"]
            vocab_size = self.tokenizer.vocab_size

            for _ in range(self.max_length):
                # Add a random token to the array
                random_token_id = random.randint(0, vocab_size-1)
                random_token = self.tokenizer.decode(random_token_id)
                encoder.append(random_token)

            encoder = " ".join(encoder)

        encoder_tokenized = self.tokenizer(encoder, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        
        dic = {
            "input_ids": encoder_tokenized["input_ids"].squeeze(0),
            "attention_mask": encoder_tokenized["attention_mask"].squeeze(0),
            "decoder_input_ids": torch.tensor([self.tokenizer.pad_token_id])
        }

        return dic