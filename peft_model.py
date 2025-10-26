from peft import (
    get_peft_model,
    LoraConfig,
    AdaLoraConfig,
    LoKrConfig,
    LoHaConfig
)
from peft import LoraConfig, get_peft_model

from utils.utils import format_num_params


INDEX_POSITIONS_VISION = {
    'openai/clip-vit-base-patch16': {
        'gradual_1': [11],
        'gradual_2': [10, 11],
        'gradual_3': [9, 10, 11],
        'gradual_4': [8, 9, 10, 11],
        'gradual_5': [7, 8, 9, 10, 11],
        'gradual_6': [6, 7, 8, 9, 10, 11],
        'gradual_7': [5, 6, 7, 8, 9, 10, 11],
        'gradual_8': [4, 5, 6, 7, 8, 9, 10, 11],
        'gradual_9': [3, 4, 5, 6, 7, 8, 9, 10, 11],
        'gradual_10': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'gradual_11': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'gradual_12': list(range(12))
    },
}


def apply_peft_to_model(
    model,
    model_name,
    peft_type: str = "none",
    use_vision_peft: bool = True,
    use_text_peft: bool = False,
    vision_config_kwargs: dict = None,
    text_config_kwargs: dict = None
):
    
    def get_target_modules(prefix, config_kwargs, model):
        base_modules = config_kwargs.get("target_modules", ["q_proj", "v_proj", "k_proj", "out_proj"])
        gradual_step = config_kwargs.get("gradual_step", None)  
        
        if prefix == "vision_model":
            num_layers = model.vision_model.config.num_hidden_layers
            
            idx_map = INDEX_POSITIONS_VISION.get(model_name, {})
            
            if gradual_step is not None:
                key = f"gradual_{gradual_step}"
                selected_layers = idx_map.get(key)
                if selected_layers is None:
                    raise ValueError(f"Gradual step {gradual_step} not in index map for {model_name}")
            else:
                selected_layers = list(range(num_layers))
                
        elif prefix == "text_model":
            num_layers = model.text_model.config.num_hidden_layers

            selected_layers = list(range(num_layers))
        else:
            raise ValueError(f"Unknown prefix: {prefix}")

        return [
            f"{prefix}.encoder.layers.{i}.self_attn.{m}"
            for i in selected_layers
            for m in base_modules
        ]
    
    for param in model.parameters():
        param.requires_grad = False
        
    target_modules = []
    modules_to_save = []
    
    if peft_type.lower() == "lora":
    
        if use_vision_peft:
            vision_config = vision_config_kwargs or {}
            target_modules.extend(get_target_modules("vision_model", vision_config, model))
            modules_to_save.extend(vision_config.get("modules_to_save", ["visual_projection"]))
        
        if use_text_peft:
            text_config = text_config_kwargs or {}
            target_modules.extend(get_target_modules("text_model", text_config, model))
            modules_to_save.extend(text_config.get("modules_to_save", ["text_projection"]))
        
        if target_modules:
      
            peft_config = LoraConfig(
                
                r=vision_config_kwargs.get("r", 8) if use_vision_peft else text_config_kwargs.get("r", 8),
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                lora_alpha=vision_config.get("lora_alpha", 4),
                lora_dropout=0.25,
                bias="none",
            )

            model = get_peft_model(model, peft_config)

            if use_vision_peft:
                model.visual_projection.original_module.weight.requires_grad_(False) 
            elif use_text_peft:
                model.text_projection.original_module.weight.requires_grad_(False)  
            
            trainable_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()

            total_params = sum(p.numel() for p in model.parameters())

            
    elif peft_type.lower() == "adalora":
        if use_vision_peft:
            vision_config = vision_config_kwargs or {}
            target_modules.extend(get_target_modules("vision_model", vision_config, model))
            modules_to_save.extend(vision_config.get("modules_to_save", ["visual_projection"]))
        if use_text_peft:
            text_config = text_config_kwargs or {}
            target_modules.extend(get_target_modules("text_model", text_config, model))
            modules_to_save.extend(text_config.get("modules_to_save", ["text_projection"]))
        if target_modules:
            peft_config = AdaLoraConfig(
                init_r=vision_config.get("init_r", 32),
                target_r=vision_config.get("r", 4),
                total_step = vision_config.get("total_step",1000),
                target_modules=target_modules,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, peft_config)
            print("[AdaLoRA] Applied.")
            
            if use_vision_peft:
                model.visual_projection.original_module.weight.requires_grad_(False) 
                print(model.visual_projection.original_module.weight.requires_grad)
            elif use_text_peft:
                model.text_projection.original_module.weight.requires_grad_(False)  
                print(model.text_projection.original_module.weight.requires_grad) 
            
            trainable_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()

            total_params = sum(p.numel() for p in model.parameters())

    elif peft_type.lower() == "loha":
        if use_vision_peft:
            vision_config = vision_config_kwargs or {}
            target_modules.extend(get_target_modules("vision_model", vision_config, model))
            modules_to_save.extend(vision_config.get("modules_to_save", ["visual_projection"]))
        if use_text_peft:
            text_config = text_config_kwargs or {}
            target_modules.extend(get_target_modules("text_model", text_config, model))
            modules_to_save.extend(text_config.get("modules_to_save", ["text_projection"]))
        if target_modules:
            peft_config = LoHaConfig(
                r=vision_config.get("r", 8),
                alpha=vision_config.get("lora_alpha", 4),
                module_dropout=vision_config.get("module_dropout", 0.1),
                target_modules=target_modules,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, peft_config)
            print("[LoHa] Applied.")
            
            if use_vision_peft:
                model.visual_projection.original_module.weight.requires_grad_(False)  
                print(model.visual_projection.original_module.weight.requires_grad)  
            elif use_text_peft:
                model.text_projection.original_module.weight.requires_grad_(False) 
                print(model.text_projection.original_module.weight.requires_grad) 
            
            trainable_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()

            total_params = sum(p.numel() for p in model.parameters())

    elif peft_type.lower() == "lokr":
        if use_vision_peft:
            vision_config = vision_config_kwargs or {}
            target_modules.extend(get_target_modules("vision_model", vision_config, model))
            modules_to_save.extend(vision_config.get("modules_to_save", ["visual_projection"]))
        if use_text_peft:
            text_config = text_config_kwargs or {}
            target_modules.extend(get_target_modules("text_model", text_config, model))
            modules_to_save.extend(text_config.get("modules_to_save", ["text_projection"]))
        if target_modules:

            peft_config = LoKrConfig(
                r=vision_config.get("r", 8),
                alpha=vision_config.get("lora_alpha", 4),
                module_dropout =vision_config.get("dropout", 0.1),
                target_modules=target_modules,
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, peft_config)
            print("[LoKr] Applied.")
            
            
            if use_vision_peft:
                model.visual_projection.original_module.weight.requires_grad_(False) 
                print(model.visual_projection.original_module.weight.requires_grad) 
            elif use_text_peft:
                model.text_projection.original_module.weight.requires_grad_(False)  
                print(model.text_projection.original_module.weight.requires_grad)
            
            trainable_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()
            total_params = sum(p.numel() for p in model.parameters())


    else:
        print(f"Unknown PEFT type: {peft_type}")
        
    return model
