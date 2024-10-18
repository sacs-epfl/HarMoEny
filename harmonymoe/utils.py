from .moe_layer import MoELayer

def replace_moe_layer(model, target, router_name, experts_name, decoder_name, config):
    _replace_moe_layer(model, target, False, [0], router_name, experts_name, decoder_name, config)

    return get_moe_layers([], model, target)

def _replace_moe_layer(model, target, is_decoder, layer_idx, router_name, experts_name, decoder_name, config):
    for name, module in model.named_children():
        if name == decoder_name:
            is_decoder = True 
            layer_idx[0] = 0

        for child_name, child in module.named_children():
            if type(child).__name__ == target:
                router = getattr(child, router_name)
                experts = getattr(child, experts_name)

                config.is_decoder = is_decoder
                config.layer_idx = layer_idx[0]
                layer_idx[0] += 1
                new_moe_layer = MoELayer(router, experts, config)

                setattr(module, child_name, new_moe_layer)
            else:
                _replace_moe_layer(
                    child, 
                    target, 
                    is_decoder,
                    layer_idx,
                    router_name, 
                    experts_name,
                    decoder_name,
                    config,
                )  

def get_moe_layers(acc, model, target):
    for module in model.children():
        if isinstance(module, MoELayer):
            acc.append(module)
        else:
            acc = get_moe_layers(acc, module, target)
    return acc