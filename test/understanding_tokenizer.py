from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8", cache_dir="/cache")

print(tokenizer("Hello world"))
print(tokenizer("Goodbye everything. see you soon"))

print(tokenizer.decode([0]))

print(tokenizer.eos_token_id)