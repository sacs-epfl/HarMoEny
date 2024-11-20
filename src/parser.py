import argparse

class ArgParse:
    def str2bool(s):
        return s.lower() in ["yes", "y", "true", "t"]

    def __init__(self):
        parser = argparse.ArgumentParser(
            prog="MoE workload generator",
            description="Spawns MoE model across GPUs and e2e iteration times",
        )

        parser.add_argument("--dataset", default="sst2", type=str)
        parser.add_argument("--num_samples", default=0, type=int, help="Number of total samples across all GPUs")
        parser.add_argument("--batch_size", default=250, type=int, help="Batch size per GPU")
        parser.add_argument("--seq_len", default=120, type=int)
        parser.add_argument("--model_name", default="google/switch-base-64", type=str, help="Huggingface model")
        parser.add_argument("--type_moe_parent", default="SwitchTransformersLayerFF", type=str, help="class name of model MoE Layer parent")
        parser.add_argument("--type_moe", default="SwitchTransformersSparseMLP", type=str, help="class name of model MoE Layers")
        parser.add_argument("--name_router", default="router", type=str, help="parameter name of router on MoE")
        parser.add_argument("--name_experts", default="experts", type=str, help="parameter name of router on MoE")
        parser.add_argument("--name_decoder", default="decoder", type=str, help="module name of model decoder")
        parser.add_argument("--dynamic_components", default=["wi", "wo"], type=list, help="parameter names of expert changing weights")
        parser.add_argument("--path", default=None, type=str, help="Specify where to save path")
        parser.add_argument("--time_dense", default=False, type=ArgParse.str2bool, help="If you want to time the dense feed-forward")
        parser.add_argument("--d_model", default=768, type=int, help="Dimension of model hidden states")

        self.parser = parser
    
    def add_argument(self, name, **kwargs):
        self.parser.add_argument(name, **kwargs)

    def parse_arguments(self):
        return self.parser.parse_args()


    # def parse_arguments():
        

    #     args, remaining_argv = parser.parse_known_args()

    #     if args.system_name != "deepspeed-inference":
    #         parser.add_argument("-w", "--world_size", default=torch.cuda.device_count(), type=int)
    #         parser.add_argument("-p", "--port", default="1234", type=str)

    #     if args.system_name == "harmony":
    #         parser.add_argument("-sched", "--scheduling_policy", default="deepspeed", type=str)
    #         parser.add_argument("-cp", "--cache_policy", default="RAND", type=str)
    #         parser.add_argument("-ec", "--expert_cache_size", default=2, type=int)
    #         parser.add_argument("-eq", "--eq_tokens", default=150, type=int)

    #     if args.system_name == "deepspeed-inference":
    #         parser.add_argument("--local_rank", default=0, type=int) 

    #     return parser.parse_args(remaining_argv, namespace=args)