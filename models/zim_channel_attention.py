import torch
import torch.nn as nn
from util.utils import get_activation

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, activation, reduction_ratio=1, residual_connection=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            get_activation(activation=activation, functional=False)(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.residual_connection = residual_connection

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y if not self.residual_connection else x + x * y
    
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 1. 定义测试参数
    in_channels = 16  # 输入通道数
    reduction_ratio = 4  # 通道缩减比例
    batch_size = 2  # 批量大小
    height, width = 32, 32  # 输入特征图尺寸
    
    # 2. 创建ChannelAttention实例（分别测试有无残差连接）
    ca_with_residual = ChannelAttention(in_channels, reduction_ratio, residual_connection=True)
    ca_without_residual = ChannelAttention(in_channels, reduction_ratio, residual_connection=False)
    
    # 3. 生成随机测试输入（形状：[batch_size, in_channels, height, width]）
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 4. 前向传播测试
    output_with_residual = ca_with_residual(x)
    output_without_residual = ca_without_residual(x)
    
    # 5. 打印测试结果信息
    print("=" * 50)
    print("ChannelAttention 测试结果")
    print("=" * 50)
    print(f"输入形状: {x.shape}")
    print(f"带残差连接的输出形状: {output_with_residual.shape}")
    print(f"无残差连接的输出形状: {output_without_residual.shape}")
    print("-" * 50)
    print(f"带残差连接的输出均值: {output_with_residual.mean():.4f}")
    print(f"无残差连接的输出均值: {output_without_residual.mean():.4f}")
    print(f"带残差连接的输出标准差: {output_with_residual.std():.4f}")
    print(f"无残差连接的输出标准差: {output_without_residual.std():.4f}")
    print("=" * 50)
    
    # 6. 验证输出形状是否正确
    assert output_with_residual.shape == x.shape, "带残差连接的输出形状与输入不匹配！"
    assert output_without_residual.shape == x.shape, "无残差连接的输出形状与输入不匹配！"
    print("测试通过：输出形状均正确！")