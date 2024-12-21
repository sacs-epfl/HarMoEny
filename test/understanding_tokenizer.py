from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8", cache_dir="/cache")

print(tokenizer("Hello world"))
print(tokenizer("Goodbye everything. see you soon"))

print(tokenizer.decode([0]))

print(tokenizer.eos_token_id)

print("__________________________________")

mixtral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", cache_dir="/cache")
print(mixtral_tokenizer("Hello"))
print(mixtral_tokenizer("World"))
print(mixtral_tokenizer("Hello World"))

print(mixtral_tokenizer.decode([0]))
print(mixtral_tokenizer.eos_token_id)