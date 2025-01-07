from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModel, AutoModelForCausalLM
import sys
import torch

with open("mixtral_quant_gptq_arch.txt", "w") as f:
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
    )

   # model = AutoGPTQForCausalLM.from_pretrained("TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", quantize_config)

    model = AutoGPTQForCausalLM.from_quantized("TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", 
        device="cpu", 
        inject_fused_attention=False, 
        inject_fused_mlp=False, 
        quantize_config=quantize_config, 
        trust_remote_code=False, 
        disable_exllama=True
    )

    # model = AutoModelForCausalLM.from_pretrained("TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    #     trust_remote_code=False,
    #     revision="main")
    sys.stdout = f
    print(model)
