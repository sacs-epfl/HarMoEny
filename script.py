import deepspeed
from deepspeed.inference import build_hf_engine 
from deepspeed import comm as dist
import torch
import os
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["HF_HOME"] = "/cache"
os.environ["TRITON_CACHE_DIR"] = "/cache/triton"

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

# Set the CUDA device for this process
torch.cuda.set_device(local_rank)
print(f"STARTING RANK: {local_rank} on GPU {local_rank}")

# Setup the distribution
dist.init_distributed(dist_backend="nccl", rank=local_rank, world_size=world_size)

tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
# moe_model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8", use_deepspeed_moe_layer=True, ep_size=8)

if tokenizer.bos_token_id is None:
    tokenizer.bos_token_id = tokenizer.pad_token_id
    # Taken from https://github.com/huggingface/transformers/issues/8161 where it says 0 == bos_token_id == pad_token_id for T5 tokenizer

from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersForConditionalGeneration
from transformers.models.switch_transformers.configuration_switch_transformers import SwitchTransformersConfig 

config = SwitchTransformersConfig(use_deepspeed_moe_layer=True, ep_size=8)
moe_model = SwitchTransformersForConditionalGeneration(config)

ds_engine = deepspeed.init_inference(moe_model, 
                                    dtype=torch.half,
                                    moe={
                                        "ep_size": 8,
                                        "moe_experts": [1], # TO BE VERIFIED: moe_experts is the number of experts you want per GPU
                                    },
                                    replace_with_kernel_inject=True)



tokens = tokenizer.encode("Yesterday a turnip arrived and wanted to deliver fantastic news to", return_tensors="pt").to(torch.cuda.current_device())
output = ds_engine.generate(tokens, max_new_tokens=5, bos_token_id=tokenizer.bos_token_id, output_router_logits=False)
if local_rank == 0:
    print(tokenizer.decode(output[0]))































# DEEPSPEED V2 does not work on V100
# from deepspeed.inference import build_hf_engine, InferenceEngineV2, RaggedInferenceEngineConfig, DeepSpeedTPConfig
# from deepspeed.inference.v2.config_v2 import QuantizationConfig
# from deepspeed.inference.v2.ragged import AllocationMode, DSStateManagerConfig, MemoryConfig

# deepspeedTPConfig = DeepSpeedTPConfig()
# deepspeedTPConfig.tp_size = 4

# quantizationConfig = QuantizationConfig()
# quantizationConfig.quantization_mode = None

# dsStateManageConfig = DSStateManagerConfig()
# dsStateManageConfig.max_tracked_sequences = 2048
# dsStateManageConfig.max_ragged_batch_size = 768
# dsStateManageConfig.max_ragged_sequence_count = 512
# dsStateManageConfig.max_context = 8192
# memoryConfig = MemoryConfig()
# memoryConfig.mode = AllocationMode.RESERVE
# memoryConfig.size = 1_000_000_000
# dsStateManageConfig.memory_config = memoryConfig
# dsStateManageConfig.offload = False

# raggedInferenceEngineConfig = RaggedInferenceEngineConfig()
# raggedInferenceEngineConfig.tensor_parallel = deepspeedTPConfig
# raggedInferenceEngineConfig.state_manager = dsStateManageConfig
# raggedInferenceEngineConfig.quantization = quantizationConfig

# inference_engine = build_hf_engine(
#     path="mistralai/Mixtral-8x7B-v0.1", 
#     engine_config=raggedInferenceEngineConfig)