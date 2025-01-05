import datasets
import torch
import random

from datasets import load_dataset
from torch.utils.data import Dataset


class FlexibleDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, model, seq_len=120, num_samples=64, random_seed=32, model_name=None, return_decoded=False, cache_dir="/cache"):
        datasets.enable_caching()

        self.tokenizer = tokenizer
        self.max_length = seq_len
        self.dataset_option = dataset_name
        self.num_samples = num_samples
        self.return_decoded = return_decoded
        self.is_switch = (model_name and "switch" in model_name)
        torch.manual_seed(random_seed)

        if self.dataset_option == "bookcorpus":
            self.dataset = load_dataset("bookcorpus/bookcorpus", split=f"train[:{num_samples}]", streaming=False, trust_remote_code=True, cache_dir=cache_dir)
            self.dataset_length = len(self.dataset)
        elif self.dataset_option == "wikitext":
            self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{num_samples}]", streaming=False, cache_dir=cache_dir)
            self.dataset_length = len(self.dataset)
        elif self.dataset_option == "sst2":
            self.dataset = load_dataset("glue", "sst2", split=f"train[:{num_samples}]", streaming=False, cache_dir=cache_dir)
            self.dataset_length = len(self.dataset)
        elif self.dataset_option == "wmt19":
            self.dataset = load_dataset("wmt/wmt19", "de-en", split=f"train[:{num_samples}]", streaming=False, cache_dir=cache_dir)
            self.dataset_length = len(self.dataset)
        elif self.dataset_option == "arxiver":
            self.dataset = load_dataset("neuralwork/arxiver", split=f"train[:{num_samples}]", streaming=False, cache_dir=cache_dir)
            self.dataset_length = len(self.dataset)
        elif self.dataset_option == "cocktail":
            num_samples_each = (num_samples // 4)+1
            self.datasets = [
                load_dataset("bookcorpus/bookcorpus", split=f"train[:{num_samples_each}]", streaming=False, trust_remote_code=True, cache_dir=cache_dir),
                load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{num_samples_each}]", streaming=False, cache_dir=cache_dir),
                load_dataset("wmt/wmt19", "de-en", split=f"train[:{num_samples_each}]", streaming=False, cache_dir=cache_dir),
                load_dataset("neuralwork/arxiver", split=f"train[:{num_samples_each}]", streaming=False, cache_dir=cache_dir),
            ]
            self.dataset_lengths = [len(self.datasets[i]) for i in range(4)]
        elif self.dataset_option == "random":
            special_ids = {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.unk_token_id
            }
            allowed_token_ids = [i for i in range(self.tokenizer.vocab_size) if i not in special_ids]
            self.pertinent_tokens = torch.tensor(allowed_token_ids)
        elif self.dataset_option == "constant":
            pass
        elif self.dataset_option.startswith("skew"):
            x = int(self.dataset_option.split("skew")[1])
            e = int((x * self.tokenizer.vocab_size - 100) / (100 - x))
            self.distribution = torch.cat([
                torch.arange(self.tokenizer.vocab_size),
                torch.zeros(e, dtype=torch.long),
            ])
            self.distribution_len = len(self.distribution)
        else:
            raise ValueError("Invalid dataset option")

    def __len__(self):
        return self.num_samples 

    def __getitem__(self, idx):
        if self.dataset_option == "bookcorpus" or self.dataset_option == "wikitext":
            text = "summarize: " + self.dataset[idx % self.dataset_length]["text"]
        elif self.dataset_option == "sst2":
            text = "summarize: " + self.dataset[idx % self.dataset_length]["sentence"]
        elif self.dataset_option == "arxiver":
            text = "summarize: " + self.dataset[idx % self.dataset_length]["abstract"]
        elif self.dataset_option == "wmt19":
            text = "translate English to German: " + self.dataset[idx % self.dataset_length]["translation"]["en"]
        elif self.dataset_option == "cocktail":
            num = random.randint(0,len(self.datasets)-1)
            data = self.datasets[num][(idx//4) % self.dataset_lengths[num]]
            if num == 0 or num == 1:
                data = data["text"]
            elif num == 2:
                data = data["translation"]["en"]
            elif num == 3:
                data = data["abstract"]
            text = f"summarize: {data}"
        elif self.dataset_option == "random":
            return self._generate_random_entry()
        elif self.dataset_option.startswith("skew"):
            return self._generate_skewed_entry()
        elif self.dataset_option == "constant":
            return self._generate_constant_entry()

        tokenized_text = self.tokenizer.encode(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        if tokenized_text.size(1) < self.max_length:
            # If the length is less than seq_len, generate random tokens to fill it
            tokens_needed = self.max_length - tokenized_text.size(1)

            random_locations = torch.randint(0, tokenized_text.shape[1], (tokens_needed,))
            random_tokens = tokenized_text[0][random_locations].unsqueeze(0)

            # Concatenate original tokens with random tokens
            before_len = tokenized_text.size(1)
            tokenized_text = torch.cat([tokenized_text, random_tokens], dim=1)

        # If the length exceeds max_length, truncate
        if tokenized_text.size(1) > self.max_length:
            tokenized_text = tokenized_text[:, :self.max_length]

        # Create the input dictionary
        encoder_tokenized = {
            "input_ids": tokenized_text.squeeze(0),
            "attention_mask": torch.full((self.max_length,), 1),
        }

        if self.is_switch:
             encoder_tokenized["decoder_input_ids"] = torch.tensor([self.tokenizer.pad_token_id])

        return encoder_tokenized
    
    def _generate_random_entry(self):
        indices = torch.randint(0, self.pertinent_tokens.shape[0], (self.max_length,))
        text_encoded = self.pertinent_tokens[indices]

        if self.return_decoded:
            return self.tokenizer.decode(text_encoded)

        encoder_tokenized = {
            "input_ids": text_encoded,
            "attention_mask": torch.full((self.max_length,), 1),
        }

        if self.is_switch:
            encoder_tokenized["decoder_input_ids"] = torch.tensor([self.tokenizer.pad_token_id])

        return encoder_tokenized
    
    def _generate_constant_entry(self):
        
        if self.is_switch:
            text_encoded = torch.full((self.max_length,), 8774) # 8774 is 'hello' in t5 tokenizer.
            text_encoded[-1] = 1
        else:
            text_encoded = torch.full((self.max_length,), 22557) # 22557 is 'hello' in mixtral
            text_encoded[0] = 1
        
        if self.return_decoded:
            return self.tokenizer.decode(text_encoded)

        encoder_tokenized = {
            "input_ids": text_encoded,
            "attention_mask": torch.full((self.max_length,), 1),
        }
        if self.is_switch:
            encoder_tokenized["decoder_input_ids"] = torch.tensor([self.tokenizer.pad_token_id])
         
        return encoder_tokenized
    
    def _generate_skewed_entry(self):
        indices = torch.randint(0, self.distribution_len, (self.max_length,), device=self.distribution.device)
        text_encoded = self.distribution[indices]

        encoder_tokenized = {
            "input_ids": text_encoded,
            "attention_mask": (text_encoded != self.tokenizer.pad_token_id).long(),
            "decoder_input_ids": torch.tensor([self.tokenizer.pad_token_id], device=self.distribution.device)
        }

        return encoder_tokenized
    
    def generate_random_batch_size_1(self):
        entry = self._generate_random_entry()
        return {
            "input_ids": torch.tensor([entry["input_ids"].tolist()]),
            "attention_mask": torch.tensor([entry["attention_mask"].tolist()]),
            "decoder_input_ids": torch.tensor([entry["decoder_input_ids"].tolist()]),
        }