import torch
import torch.nn as nn
from diffusion_policy.model.RT1.robotic_transformer_pytorch import MaxViT, RT1

loss_object = nn.CrossEntropyLoss(reduction='none')
vit = MaxViT(
    num_classes = 1000,
    dim_conv_stem = 64,
    dim = 96,
    dim_head = 32,
    depth = (2, 5, 2),
    window_size = 4,
    mbconv_expansion_rate = 4,
    mbconv_shrinkage_rate = 0.25,
    dropout = 0.1
)

model = RT1(
    vit = vit,
    num_actions = 11,
    depth = 6,
    heads = 8,
    dim_head = 64,
    cond_drop_prob = 0.2
)

def print_shapes(module, input, output):
    print(f"{module.__class__.__name__} input shape: {input[0].shape}, output shape: {output.shape}")

for layer in model.children():
    layer.register_forward_hook(print_shapes)

video = torch.randn(2, 3, 6, 64, 64)

instructions = [
    'bring me that apple sitting on the table',
    'please pass the butter'
]

instructions = None

train_logits = model(video, instructions) # (2, 6, 11, 256) # (batch, frames, actions, bins)
action_tokens = torch.randint(0, 11, (2, 6, 11))
print(torch.mean(loss_object(train_logits.permute(0, 3, 1, 2), action_tokens), dim=-1).shape)
# after much training

# model.eval()
# eval_logits = model(video, instructions, cond_scale = 3.) # classifier free guidance with conditional scale of 3
# print(eval_logits.shape)