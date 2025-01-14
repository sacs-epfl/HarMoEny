import argparse
import torch


def str2bool(s):
    return s.lower() in ["yes", "y", "true", "t"]

class Args:
    def harmony(self):
        parser = argparse.ArgumentParser(
            prog="MoE workload generator",
            description="Spawns MoE model across GPUs and e2e iteration times",
        )
        parser.add_argument("--dataset", default="sst2", type=str)
        parser.add_argument(
            "--num_samples",
            default=0,
            type=int,
            help="Number of total samples across all GPUs",
        )
        parser.add_argument(
            "--batch_size", default=None, type=int, help="Batch size per GPU"
        )
        parser.add_argument("--start_batch_size", default=1, type=int)
        parser.add_argument("--seq_len", default=120, type=int)
        parser.add_argument("--num_experts", default=8, type=int)
        parser.add_argument(
            "--loader",
            default="transformers",
            type=str,
            help="System to load model",
        )
        parser.add_argument(
            "--model_name",
            default="google/switch-base-64",
            type=str,
            help="Huggingface model",
        )
        parser.add_argument(
            "--type_moe_parent",
            default="SwitchTransformersLayerFF",
            type=str,
            help="class name of model MoE Layer parent",
        )
        parser.add_argument(
            "--type_moe",
            default="SwitchTransformersSparseMLP",
            type=str,
            help="class name of model MoE Layers",
        )
        parser.add_argument(
            "--router_tensor_path",
            default="router.classifier",
            type=str,
            help="path from MoE layer to router's tensor",
        )
        parser.add_argument(
            "--name_experts",
            default="experts",
            type=str,
            help="parameter name of router on MoE",
        )
        parser.add_argument(
            "--name_decoder",
            default="decoder",
            type=str,
            help="module name of model decoder",
        )
        parser.add_argument(
            "--d_model", default=768, type=int, help="Dimension of model hidden states"
        )
        parser.add_argument(
            "--model_dtype", type=str, help="float, float16, or int8"
        )
        parser.add_argument("--scheduling_policy", default="deepspeed", type=str)
        parser.add_argument("--cache_policy", default="RAND", type=str)
        parser.add_argument("--expert_cache_size", default=2, type=int)
        parser.add_argument("--eq_tokens", default=150, type=int)
        parser.add_argument("--world_size", default=torch.cuda.device_count(), type=int)
        parser.add_argument("--port", default="1234", type=str)
        parser.add_argument("--warmup_rounds", default=3, type=int)
        parser.add_argument("--path", default="outputs/harmony", type=str)
        parser.add_argument("--expert_placement", default=None, type=str)
        parser.add_argument("--enable_router_skew", default=False, type=str2bool)
        parser.add_argument("--enable_router_random", default=False, type=str2bool)
        parser.add_argument("--enable_router_uniform", default=False, type=str2bool)
        parser.add_argument(
            "--router_skew", default=0.0, type=float, help="Value between 0 and 1"
        )
        parser.add_argument(
            "--router_num_experts_skewed",
            default=1,
            type=int,
            help="Number of experts that receive the skewed proportion",
        )
        parser.add_argument(
            "--random_router_skew",
            default=False,
            type=str2bool,
            help="Whether to enable random skewing in the router",
        )
        parser.add_argument(
            "--expert_fetching_strategy",
            default="async-cpu",
            type=str,
            help="The kind of expert fetching desired",
        )

        return parser.parse_args()
    
    def fastmoe(self):
        parser = argparse.ArgumentParser(
            prog="Run inference on FasterMoE",
        )
        parser.add_argument(
            "--model_name",
            default="google/switch-base-64",
            type=str,
            help="Huggingface model",
        )
        parser.add_argument(
            "--d_model", default=768, type=int, help="Dimension of model hidden states"
        )
        parser.add_argument(
            "--type_moe_parent",
            default="SwitchTransformersLayerFF",
            type=str,
            help="class name of model MoE Layer parent",
        )
        parser.add_argument(
            "--type_moe",
            default="SwitchTransformersSparseMLP",
            type=str,
            help="class name of model MoE Layers",
        )
        parser.add_argument(
            "--router_tensor_path",
            default="router.classifier",
            type=str,
            help="path from MoE layer to router's tensor",
        )
        parser.add_argument(
            "--name_experts",
            default="experts",
            type=str,
            help="parameter name of router on MoE",
        )
        parser.add_argument("--system_name", default="fastmoe", type=str)
        parser.add_argument("--dataset", default="sst2", type=str)
        parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
        parser.add_argument("--batch_size", default=250, type=int, help="Batch size per GPU")
        parser.add_argument("--seq_len", default=120, type=int)
        parser.add_argument("--path", default="outputs/out", type=str, help="Specify where to save path")
        parser.add_argument("--num_experts", default=8, type=int, help="Number of experts we want to match dense model to")
        parser.add_argument("--world_size", default=torch.cuda.device_count(), type=int, help="Number of GPUs to use")
        parser.add_argument("--port", default="1234", type=str)
        parser.add_argument("--warmup_rounds", default=3, type=int)
        parser.add_argument("--random_router_skew", default=False, type=str2bool, help="Wether to enable random skewing in the router")
        parser.add_argument("--enable_router_skew", default=False, type=str2bool)
        parser.add_argument("--enable_router_random", default=False, type=str2bool)
        parser.add_argument("--enable_router_uniform", default=False, type=str2bool)
        parser.add_argument(
            "--router_skew", default=0.0, type=float, help="Value between 0 and 1"
        )
        parser.add_argument(
            "--router_num_experts_skewed",
            default=1,
            type=int,
            help="Number of experts that receive the skewed proportion",
        )

        return parser.parse_args()
    
    def deepspeed(self):
        parser = argparse.ArgumentParser(
            prog="Run inference on DeepSpeed-MoE inference engine",
        )
        parser.add_argument(
            "--model_name",
            default="google/switch-base-64",
            type=str,
            help="Huggingface model",
        )
        parser.add_argument(
            "--d_model", default=768, type=int, help="Dimension of model hidden states"
        )
        parser.add_argument(
            "--type_moe_parent",
            default="SwitchTransformersLayerFF",
            type=str,
            help="class name of model MoE Layer parent",
        )
        parser.add_argument(
            "--type_moe",
            default="SwitchTransformersSparseMLP",
            type=str,
            help="class name of model MoE Layers",
        )
        parser.add_argument(
            "--router_tensor_path",
            default="router.classifier",
            type=str,
            help="path from MoE layer to router's tensor",
        )
        parser.add_argument(
            "--name_experts",
            default="experts",
            type=str,
            help="parameter name of router on MoE",
        )
        parser.add_argument("--dataset", default="sst2", type=str)
        parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
        parser.add_argument("--batch_size", default=250, type=int, help="Batch size per GPU")
        parser.add_argument("--seq_len", default=120, type=int)
        parser.add_argument("--path", default="outputs/out", type=str, help="Specify where to save path")
        parser.add_argument("--num_experts", default=8, type=int, help="Number of experts we want to match dense model to")
        parser.add_argument("--warmup_rounds", default=3, type=int)
        parser.add_argument("--local_rank", default=0, type=int) 
        parser.add_argument("--world_size", default=8, type=int)
        parser.add_argument("--capacity_factor", default=10.0, type=float)
        parser.add_argument("--random_router_skew", default=False, type=str2bool, help="Wether to enable random skewing in the router")
        parser.add_argument("--enable_router_skew", default=False, type=str2bool)
        parser.add_argument("--enable_router_random", default=False, type=str2bool)
        parser.add_argument("--enable_router_uniform", default=False, type=str2bool)
        parser.add_argument(
            "--router_skew", default=0.0, type=float, help="Value between 0 and 1"
        )
        parser.add_argument(
            "--router_num_experts_skewed",
            default=1,
            type=int,
            help="Number of experts that receive the skewed proportion",
        )

        return parser.parse_args()
