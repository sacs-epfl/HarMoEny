from transformers import AutoModel 

from harmonymoe.utils import replace_moe_layer
from harmonymoe.moe_layer import MoEConfig, MoELayer

from .utils import replace_fmoe_layer

class Modeling:
    def __init__(self, model_name="google/switch-base-64", system_name="harmony", scheduling_policy="deepspeed", cache_policy="RAND", expert_cache_size=1, dynamic_components=[], eq_tokens=150, name_moe_layer="", name_router="", name_experts="", name_decoder="", **kwargs):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, cache_dir="/cache")
        self.config = self.model.config
        self.system_name = system_name

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
        elif system_name == "fastmoe":
            self.moe_layers = replace_fmoe_layer(
                self.model,
                name_moe_layer,
                name_router,
                name_experts,
                name_decoder,
                self.config,
            )
        else:
            raise Exception("This sytem is not implemented")
    
    def __call__(self, batch):
      #  if self.system_name == "fastme"

        if "switch" in self.model_name:
            # if self.system_name == "fastmoe":
            #     print(list(map(lambda d: f"{d[0]}: {d[1].shape}", batch.items())))
            #     batch["input_ids"] = batch["input_ids"].reshape(-1)
            #     batch["attention_mask"] = batch["attention_mask"].reshape(-1)
            #     batch["decoder_input_ids"] = batch["decoder_input_ids"].reshape(-1)
            #     print(list(map(lambda d: f"{d[0]}: {d[1].shape}", batch.items())))
            #     exit(0)
            #     # FastMoE expects the second dimension to be d_model
            #     inp = inp.reshape(-1, self.d_model)
            # print(list(map(lambda d: f"{d[0]}: {d[1].shape}", batch.items())))
            # exit(0)
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
                if not isinstance(layer, MoELayer):
                    break
                layer.save_statistics(DIR=path)
