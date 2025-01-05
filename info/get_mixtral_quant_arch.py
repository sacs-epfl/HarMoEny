# https://huggingface.co/hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4
# AutoAWQ
import sys
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM

with open("mixtral_quant_arch.txt", "w") as f:
    model_id = "hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4"
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    sys.stdout = f
    print(model)


### TRANFORMER ####
# import sys

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

# model_id = "hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4"

# quantization_config = AwqConfig(
#     bits=4,
#     fuse_max_seq_len=512, # Note: Update this as per your use-case
#     do_fuse=True,
# )

# with open("mixtral_quant_arch.txt", "w") as f:
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.float16,
#         low_cpu_mem_usage=True,
#         device_map="auto",
#         quantization_config=quantization_config
#     )    
#     sys.stdout = f
#     print(model)

### THE BLOKE ####
# from transformers import AutoModelForCausalLM

# with open("mixtral_quant_arch.txt", "w") as f:
#     model = AutoModelForCausalLM.from_pretrained("TheBloke/Mixtral-8x7B-v0.1-GPTQ", cache_dir="/cache", device_map="auto", trust_remote_code=False, revision="main")
#     sys.stdout = f
#     print(model)
