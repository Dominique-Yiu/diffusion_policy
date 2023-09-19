import torch
from torchvision import transforms
from einops import rearrange, pack, unpack

# 假设你有一个输入张量 input_data，其形状为 [batch_size, frames, height, width, channels]
# 和目标调整后的尺寸 resize_height 和 resize_width

# 定义 batch_size, frames, height, width, channels
batch_size = 4
frames = 10
height = 128
width = 128
channels = 3

# 定义目标调整后的尺寸
resize_height = 64
resize_width = 64

# 创建随机输入数据（这里只是示例，实际应该使用你的数据）
input_data = torch.randn(batch_size, frames, height, width, channels)
input_data = rearrange(input_data, 'batch_size frames height width channels -> (batch_size frames) channels height width')
print(input_data.shape)
# 创建变换组合
sample = torch.randn(input_data.shape)
print(f'sample shape: {sample.shape}')
transform = transforms.Compose([
    transforms.Resize((resize_height, resize_width))
])
print(transform)
sample_T = transform(sample)
print(f'sample_T shape {sample_T.shape}')
# 对输入数据进行变换
resized_data = transform(input_data)

# 输出调整后的张量形状
print("Resized Data Shape:", resized_data.shape)
restored_data = rearrange(resized_data, '(batch_size frames) channels height width -> batch_size frames height width channels', batch_size=batch_size)
print(f"Restore size: {restored_data.shape}")