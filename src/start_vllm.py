# GETTING CONFIG #
# from vllm.transformers_utils.config import get_config
# config = get_config("mistralai/Mixtral-8x7B-Instruct-v0.1", trust_remote_code=True)
# print(config)
########################

# from vllm.platform import current_platform
# print(current_platform.get_device_name())


from vllm import LLM

try:
    llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1", task="generate", tensor_parallel_size=8, trust_remote_code=True)
    output = llm.generate("Hello, my name is")
    print(output)
finally:
    destroy_process_group()

