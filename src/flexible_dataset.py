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
            self.dataset = load_dataset("neuralwork/arxiver", split=f"train[:{num_samples}]", streaming=False, cache_dir="/cache")
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
            text = "summarize: " + self.dataset[idx]["text"]
        elif self.dataset_option == "sst2":
            text = "summarize: " + self.dataset[idx]["sentence"]
        elif self.dataset_option == "arxiver":
            text = "summarize: " + self.dataset[idx]["abstract"]
        elif self.dataset_option == "wmt19":
            text = "translate English to German: " + self.dataset[idx]["translation"]["en"]
        elif self.dataset_option == "random":
            return self._generate_random_entry()
            # encoder = ["summarize:"]
            # vocab_size = self.tokenizer.vocab_size

            # for _ in range(self.max_length):
            #     # Add a random token to the array
            #     random_token_id = random.randint(0, vocab_size-1)
            #     random_token = self.tokenizer.decode(random_token_id)
            #     encoder.append(random_token)

            # encoder = " ".join(encoder)

        # encoder_tokenized = self.tokenizer(encoder, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # dic = {
        #     "input_ids": encoder_tokenized["input_ids"].squeeze(0),
        #     "attention_mask": encoder_tokenized["attention_mask"].squeeze(0),
        #     "decoder_input_ids": torch.tensor([self.tokenizer.pad_token_id])
        # }

        tokenized_text = self.tokenizer.encode(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        if tokenized_text.size(1) < self.max_length:
            # If the length is less than seq_len, generate random tokens to fill it
            #random_tokens = []
            #vocab_size = self.tokenizer.vocab_size
            tokens_needed = self.max_length - tokenized_text.size(1)

            random_locations = torch.randint(0, tokenized_text.shape[1], (tokens_needed,))
            #print(random_locations.shape)
            random_tokens = tokenized_text[0][random_locations].unsqueeze(0)
          
            #random_token_ids = self.tokenizer.convert_tokens_to_ids(random_tokens)

            # for _ in range(tokens_needed):
            #     random_token_id = random.randint(0, vocab_size - 1)
            #     random_tokens.append(random_token_id)

            # Concatenate original tokens with random tokens
            before_len = tokenized_text.size(1)
            #tokenized_text = torch.cat([tokenized_text, torch.tensor(random_tokens).unsqueeze(0)], dim=1)
            tokenized_text = torch.cat([tokenized_text, random_tokens], dim=1)
            #print(f"before:{before_len} after:{tokenized_text.size(1)}")

        # If the length exceeds max_length, truncate
        if tokenized_text.size(1) > self.max_length:
            tokenized_text = tokenized_text[:, :self.max_length]

        # Create the input dictionary
        encoder_tokenized = {
            "input_ids": tokenized_text.squeeze(0),
            "attention_mask": (tokenized_text != self.tokenizer.pad_token_id).long().squeeze(0),
            "decoder_input_ids": torch.tensor([self.tokenizer.pad_token_id])
        }

        # print(f"input_ids: {encoder_tokenized['input_ids'].shape} {encoder_tokenized['input_ids'].dtype}")
        # print(f"attention_mask: {encoder_tokenized['attention_mask'].shape} {encoder_tokenized['attention_mask'].dtype}")
        # print(f"decoder_input_ids: {encoder_tokenized['decoder_input_ids'].shape} {encoder_tokenized['decoder_input_ids'].dtype}")

        return encoder_tokenized
    
    def _generate_random_entry(self):
        # encoder = ["summarize:"]
        # text_encoded = self.tokenizer(encoder, return_tensors="pt")

        text_encoded = torch.randint(0, self.tokenizer.vocab_size, (self.max_length,))

        # rand = []

        # vocab_size = self.tokenizer.vocab_size
        # for _ in range(self.max_length):
        #     random_token_id = random.randint(0, vocab_size - 1)
        #     rand.append(random_token_id)

        # vocab_size = self.tokenizer.vocab_size

        # for _ in range(self.max_length):
        #     # Add a random token to the array
        #     random_token_id = random.randint(0, vocab_size - 1)
        #     random_token = self.tokenizer.decode(random_token_id)
        #     encoder.append(random_token)

        # encoder = " ".join(encoder)
        # encoder_tokenized = self.tokenizer(encoder, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        encoder_tokenized = {
            "input_ids": text_encoded,
            "attention_mask": (text_encoded != self.tokenizer.pad_token_id).long(),
            "decoder_input_ids": torch.tensor([self.tokenizer.pad_token_id])
        }

        # print(f"input_ids: {encoder_tokenized['input_ids'].shape} {encoder_tokenized['input_ids'].dtype}")
        # print(f"attention_mask: {encoder_tokenized['attention_mask'].shape} {encoder_tokenized['attention_mask'].dtype}")
        # print(f"decoder_input_ids: {encoder_tokenized['decoder_input_ids'].shape} {encoder_tokenized['decoder_input_ids'].dtype}")

        return encoder_tokenized