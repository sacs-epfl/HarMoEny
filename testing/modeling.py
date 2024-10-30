import os 

from transformers import AutoModel 

from harmonymoe.utils import replace_moe_layer
from harmonymoe.moe_layer import MoEConfig, MoELayer

from .utils import replace_fmoe_layer, replace_deepspeed_layer, replace_dense_layer

class Modeling:
    def __init__(self, model_name="google/switch-base-64", system_name="harmony", scheduling_policy="deepspeed", cache_policy="RAND", expert_cache_size=1, dynamic_components=[], eq_tokens=150, name_moe_layer="", name_router="", name_experts="", name_decoder="", time_dense=False, name_dense="SwitchTransformersLayerFF", batch_size=1000, seq_len=120, **kwargs):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, cache_dir="/cache")
        self.config = self.model.config
        self.system_name = system_name
        self.num_experts = self.model.config.num_experts

        if system_name == "harmony":
            config = MoEConfig(
                scheduling_policy=scheduling_policy,
                cache_policy=cache_policy,
                expert_cache_size=expert_cache_size,
                dynamic_components=dynamic_components,
                eq_tokens=eq_tokens,
                d_model=self.config.d_model,
            )
            self.moe_layers = replace_moe_layer(
                self.model, 
                name_moe_layer, 
                name_router, 
                name_experts,
                name_decoder,
                config,
            )
        elif system_name == "fastmoe" or system_name == "fastermoe":
            if system_name == "fastermoe":
                os.environ["FMOE_FASTER_SCHEDULE_ENABLE"] = "1"
                os.environ["FMOE_FASTER_GROUP_SIZE"] = "8"
                os.environ["FMOE_FASTER_SHADOW_ENABLE"] = "1"
                os.environ["FMOE_FASTER_GLBPLC_ALPHA"] = "4"
                os.environ["FMOE_FASTER_GLBPLC_GPUTP"] = "13e12"
                os.environ["FMOE_FASTER_GLBPLC_NETBW"] = "22.46"

            self.moe_layers = replace_fmoe_layer(
                self.model,
                name_moe_layer,
                name_router,
                name_experts,
                name_decoder,
                self.config,
            )
        elif system_name == "deepspeed-inference":
            self.moe_layers = replace_deepspeed_layer(
                self.model,
                name_moe_layer,
                name_router,
                name_experts,
                name_decoder,
                self.config,
            )
        else:
            raise Exception("This sytem is not implemented")


        if time_dense:
            self.dense = replace_dense_layer(
                self.model,
                name_dense,
                name_decoder,
                self.config,
            )
    
    def __call__(self, batch):
        if "switch" in self.model_name:
            return self.model(**batch)
        else:
            raise Exception("This model has not been implemented")
    
    def cuda(self):
        self.model.cuda()
    
    def eval(self):
        self.model.eval()
    
    def save_statistics(self, path):
        if self.moe_layers:
            for layer in self.moe_layers:
                layer.save_statistics(DIR=path)
        if self.dense:
            for layer in self.dense:
                layer.save_statistics(DIR=path)
    
    def available_systems():
        return ["deepspeed-inference", "harmony", "fastmoe", "fastermoe"]