from transformers import AutoModel 

from harmonymoe.utils import replace_moe_layer
from harmonymoe.moe_layer import MoEConfig

class Modelling:
    def __init__(self, model_name="google/switch-base-64", system_name="harmony", scheduling_policy="deepspeed", cache_policy="RAND", expert_cache_size=1, dynamic_components=[], eq_tokens=150, d_model=768, name_moe_layer="", name_router="", name_experts="", name_decoder="", **kwargs):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, cache_dir="/cache")
        config = MoEConfig(
            scheduling_policy=scheduling_policy,
            cache_policy=cache_policy,
            expert_cache_size=expert_cache_size,
            dynamic_components=dynamic_components,
            eq_tokens=eq_tokens,
            d_model=d_model,
        )
        self.moe_layers = replace_moe_layer(
            self.model, 
            name_moe_layer, 
            name_router, 
            name_experts,
            name_decoder,
            config,
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
