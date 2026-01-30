import torch
import numpy as np

def create_optimizer(model, config):
    train_config = config['training']
    base_lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    
    layer_decay = train_config.get('layer_decay', 0.8) 
    
    # 获取所有的 blocks 数量用于计算深度
    # 假设 model 是 HieraClassifier，其 encoder blocks 在 self.blocks 中
    num_layers = len(model.blocks) + 1 # +1 处理 patch_embed
    
    parameter_groups = []
    
    # 1. 专门处理 Head (分类头通常使用最大的 base_lr)
    head_lr = train_config.get('head_lr', base_lr)
    parameter_groups.append({
        "params": [p for n, p in model.named_parameters() if "head" in n],
        "lr": head_lr,
        "weight_decay": weight_decay
    })

    # 2. 处理 Encoder Blocks (按层衰减)
    for i, block in enumerate(model.blocks):
        # 深度越深（靠近 head），学习率越高
        # 最后一层 i = num_layers-2，缩放接近 1.0
        # 第一层 i = 0，缩放为 layer_decay^(num_layers)
        scale = layer_decay ** (num_layers - i - 1)
        
        parameter_groups.append({
            "params": block.parameters(),
            "lr": base_lr * scale,
            "weight_decay": weight_decay
        })

    # 3. 处理 Patch Embed 和其他初始层 (最低的学习率)
    earliest_params = []
    for n, p in model.named_parameters():
        if "patch_embed" in n or "encoder_norm" in n:
            earliest_params.append(p)
    
    if earliest_params:
        parameter_groups.append({
            "params": earliest_params,
            "lr": base_lr * (layer_decay ** num_layers),
            "weight_decay": weight_decay
        })

    if train_config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=tuple(train_config['betas']),
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            parameter_groups,
            momentum=train_config.get('momentum', 0.9),
            weight_decay=train_config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")

    return optimizer


def create_lr_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler"""
    train_config = config['training']
    total_steps = train_config['epochs'] * steps_per_epoch
    warmup_steps = train_config['warmup_epochs'] * steps_per_epoch

    if train_config['lr_scheduler'].lower() == 'cosine':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(train_config['min_lr'] / train_config['learning_rate'],
                          0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {train_config['lr_scheduler']}")

    return scheduler