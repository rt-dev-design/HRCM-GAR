import torch
import torch.nn as nn
import torch.nn.functional as F
from util.utils import get_activation, HyperParameters


class LearnedQueryAttentionFeatureAggregator(nn.Module, HyperParameters):
    """
    Multihead attention feature aggregation module.

    Inputs:
        - x: Tensor of shape (B, X, Y, D)
    Outputs:
        - aggregated: Tensor of shape (B, D)
    """

    def __init__(self, 
        dim, nhead, aggregation_dim, 
        activation, use_ffn=False, dropout=0.1, 
        use_norm=False, norm_first=False,
        num_queries=1
    ):
        super(LearnedQueryAttentionFeatureAggregator, self).__init__()
        self.save_hyperparameters()
        assert self.aggregation_dim == [1, 2] or self.aggregation_dim == [1], "aggregation_dim must be either [1, 2] or [1]"
        self.query_embed = nn.Embedding(self.num_queries, self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.nhead, batch_first=True)
        self.dropout_after_attn = nn.Dropout(self.dropout)
        if self.use_ffn: 
            self.ffn = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                get_activation(activation=self.activation, functional=False)(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim * 4, self.dim),
            )
        if self.use_norm: 
            self.norm = nn.LayerNorm(self.dim)

    def forward(self, x):
        if self.aggregation_dim == [1, 2]:
            assert len(x.shape) == 4, "Expected x to be of shape (B, X, Y, D)"
            B, X, Y, D = x.shape
            x = x.reshape(B, X * Y, D)
        elif self.aggregation_dim == [1]:
            assert len(x.shape) == 3, "Expected x to be of shape (B, X, D)"
            B, X, D = x.shape
        else:
            raise ValueError("aggregation_dim should be either [1, 2] or [1]")
        assert D == self.dim, "Expected last dim to be of length {} but got {}".format(self.dim, D)

        # Prepare learnable query -> (B, num_queries, D)
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        if self.use_norm and self.norm_first:
            query = self.norm(query)
        # Multihead attention: queries attend to sequence
        # query: (B, num_queries, D), key/value: (B, N, D)
        out, attn = self.attn(query, x, x)
        out = self.dropout_after_attn(out)
        if self.use_ffn: 
            out = self.ffn(out)
        if self.use_norm and not self.norm_first: 
            out = self.norm(out)
        # out: (B, 1, D) -> (B, D), or no ops for more than 1 query
        return torch.squeeze(out), attn

class LearnedQueryAttentionV0(nn.Module, HyperParameters):
    """
    Multihead attention feature aggregation module.

    Inputs:
        - x: Tensor of shape (B, X, Y, D)
    Outputs:
        - aggregated: Tensor of shape (B, D)
    """

    def __init__(self, 
        aggregation_dim,
        dim, nhead,  
        activation, use_ffn=False, dropout=0.1,
        num_queries=1
    ):
        super(LearnedQueryAttentionV0, self).__init__()
        self.save_hyperparameters()
        assert self.aggregation_dim == [1, 2] or self.aggregation_dim == [1], "aggregation_dim must be either [1, 2] or [1]"
        
        self.query_embed = nn.Embedding(self.num_queries, self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.nhead, batch_first=True)
        self.dropout_after_attn = nn.Dropout(self.dropout)
        
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                get_activation(activation=self.activation, functional=False)(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim * 4, self.dim)
            ) 

    def forward(self, x):
        if self.aggregation_dim == [1, 2]:
            assert len(x.shape) == 4, "Expected x to be of shape (B, X, Y, D)"
            B, X, Y, D = x.shape
            x = x.reshape(B, X * Y, D)
        elif self.aggregation_dim == [1]:
            assert len(x.shape) == 3, "Expected x to be of shape (B, X, D)"
            B, X, D = x.shape
        else:
            raise ValueError("aggregation_dim should be either [1, 2] or [1]")
        assert D == self.dim, "Expected last dim to be of length {} but got {}".format(self.dim, D)

        input_query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        extracted, attn = self.attn(input_query, x, x)
        enriched_query = self.dropout_after_attn(extracted)

        if self.use_ffn:
            output = self.ffn(enriched_query)
        else:
            output = enriched_query
        # out: (B, 1, D) -> (B, D), or no ops for more than 1 query
        return torch.squeeze(output), attn

class LearnedQueryAttentionV1(nn.Module, HyperParameters):
    """
    Multihead attention feature aggregation module.

    Inputs:
        - x: Tensor of shape (B, X, Y, D)
    Outputs:
        - aggregated: Tensor of shape (B, D)
    """

    def __init__(self, 
        aggregation_dim,
        dim, nhead,  
        activation, use_ffn=False, dropout=0.1,
        num_queries=1
    ):
        super(LearnedQueryAttentionV1, self).__init__()
        self.save_hyperparameters()
        assert self.aggregation_dim == [1, 2] or self.aggregation_dim == [1], "aggregation_dim must be either [1, 2] or [1]"
        
        self.query_embed = nn.Embedding(self.num_queries, self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.nhead, batch_first=True)
        self.dropout_after_attn = nn.Dropout(self.dropout)
        
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                get_activation(activation=self.activation, functional=False)(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim * 4, self.dim),
                nn.Dropout(self.dropout)
            ) 

    def forward(self, x):
        if self.aggregation_dim == [1, 2]:
            assert len(x.shape) == 4, "Expected x to be of shape (B, X, Y, D)"
            B, X, Y, D = x.shape
            x = x.reshape(B, X * Y, D)
        elif self.aggregation_dim == [1]:
            assert len(x.shape) == 3, "Expected x to be of shape (B, X, D)"
            B, X, D = x.shape
        else:
            raise ValueError("aggregation_dim should be either [1, 2] or [1]")
        assert D == self.dim, "Expected last dim to be of length {} but got {}".format(self.dim, D)

        input_query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        extracted, attn = self.attn(input_query, x, x)
        enriched_query = self.dropout_after_attn(extracted)

        if self.use_ffn:
            output = self.ffn(enriched_query)
        else:
            output = enriched_query
        # out: (B, 1, D) -> (B, D), or no ops for more than 1 query
        return torch.squeeze(output), attn

class LearnedQueryAttentionV2(nn.Module, HyperParameters):
    """
    Multihead attention feature aggregation module.

    Inputs:
        - x: Tensor of shape (B, X, Y, D)
    Outputs:
        - aggregated: Tensor of shape (B, D)
    """

    def __init__(self, 
        aggregation_dim,
        dim, nhead,  
        activation, use_ffn=False, dropout=0.1,
        num_queries=1
    ):
        super(LearnedQueryAttentionV2, self).__init__()
        self.save_hyperparameters()
        assert self.aggregation_dim == [1, 2] or self.aggregation_dim == [1], "aggregation_dim must be either [1, 2] or [1]"
        
        self.query_embed = nn.Embedding(self.num_queries, self.dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.nhead, batch_first=True)
        self.dropout_after_attn = nn.Dropout(self.dropout)
        
        if self.use_ffn: 
            self.norm2 = nn.LayerNorm(self.dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                get_activation(activation=self.activation, functional=False)(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim * 4, self.dim),
                nn.Dropout(self.dropout)
            ) 

    def forward(self, x):
        if self.aggregation_dim == [1, 2]:
            assert len(x.shape) == 4, "Expected x to be of shape (B, X, Y, D)"
            B, X, Y, D = x.shape
            x = x.reshape(B, X * Y, D)
        elif self.aggregation_dim == [1]:
            assert len(x.shape) == 3, "Expected x to be of shape (B, X, D)"
            B, X, D = x.shape
        else:
            raise ValueError("aggregation_dim should be either [1, 2] or [1]")
        assert D == self.dim, "Expected last dim to be of length {} but got {}".format(self.dim, D)

        input_query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        normed_query = self.norm1(input_query)
        extracted, attn = self.attn(normed_query, x, x)
        enriched_query = input_query + self.dropout_after_attn(extracted)

        if self.use_ffn:
            output = enriched_query + self.ffn(self.norm2(enriched_query))
        else:
            output = enriched_query
        # out: (B, 1, D) -> (B, D), or no ops for more than 1 query
        return torch.squeeze(output), attn

class LearnedQueryAttentionV3(nn.Module, HyperParameters):
    """
    Multihead attention feature aggregation module.

    Inputs:
        - x: Tensor of shape (B, X, Y, D)
    Outputs:
        - aggregated: Tensor of shape (B, D)
    """

    def __init__(self, 
        aggregation_dim,
        dim, nhead,  
        activation, use_ffn=False, dropout=0.1,
        num_queries=1
    ):
        super(LearnedQueryAttentionV3, self).__init__()
        self.save_hyperparameters()
        assert self.aggregation_dim == [1, 2] or self.aggregation_dim == [1], "aggregation_dim must be either [1, 2] or [1]"
        
        self.query_embed = nn.Embedding(self.num_queries, self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.nhead, batch_first=True)
        self.dropout_after_attn = nn.Dropout(self.dropout)
        
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                get_activation(activation=self.activation, functional=False)(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim * 4, self.dim),
                nn.Dropout(self.dropout)
            ) 

    def forward(self, x):
        if self.aggregation_dim == [1, 2]:
            assert len(x.shape) == 4, "Expected x to be of shape (B, X, Y, D)"
            B, X, Y, D = x.shape
            x = x.reshape(B, X * Y, D)
        elif self.aggregation_dim == [1]:
            assert len(x.shape) == 3, "Expected x to be of shape (B, X, D)"
            B, X, D = x.shape
        else:
            raise ValueError("aggregation_dim should be either [1, 2] or [1]")
        assert D == self.dim, "Expected last dim to be of length {} but got {}".format(self.dim, D)

        input_query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        extracted, attn = self.attn(input_query, x, x)
        enriched_query = input_query + self.dropout_after_attn(extracted)

        if self.use_ffn:
            output = enriched_query + self.ffn(enriched_query)
        else:
            output = enriched_query
        # out: (B, 1, D) -> (B, D), or no ops for more than 1 query
        return torch.squeeze(output), attn

class LearnedQueryAttentionV4(nn.Module, HyperParameters):
    """
    Multihead attention feature aggregation module.

    Inputs:
        - x: Tensor of shape (B, X, Y, D)
    Outputs:
        - aggregated: Tensor of shape (B, D)
    """

    def __init__(self, 
        aggregation_dim,
        dim, nhead,  
        activation, use_ffn=False, dropout=0.1,
        num_queries=1
    ):
        super(LearnedQueryAttentionV4, self).__init__()
        self.save_hyperparameters()
        assert self.aggregation_dim == [1, 2] or self.aggregation_dim == [1], "aggregation_dim must be either [1, 2] or [1]"
        
        self.query_embed = nn.Embedding(self.num_queries, self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.nhead, batch_first=True)
        self.dropout_after_attn = nn.Dropout(self.dropout)
        
        if self.use_ffn:
            self.norm = nn.LayerNorm(self.dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                get_activation(activation=self.activation, functional=False)(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim * 4, self.dim),
                nn.Dropout(self.dropout)
            ) 

    def forward(self, x):
        if self.aggregation_dim == [1, 2]:
            assert len(x.shape) == 4, "Expected x to be of shape (B, X, Y, D)"
            B, X, Y, D = x.shape
            x = x.reshape(B, X * Y, D)
        elif self.aggregation_dim == [1]:
            assert len(x.shape) == 3, "Expected x to be of shape (B, X, D)"
            B, X, D = x.shape
        else:
            raise ValueError("aggregation_dim should be either [1, 2] or [1]")
        assert D == self.dim, "Expected last dim to be of length {} but got {}".format(self.dim, D)

        input_query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        extracted, attn = self.attn(input_query, x, x)
        enriched_query = input_query + self.dropout_after_attn(extracted)

        if self.use_ffn:
            output = enriched_query + self.ffn(self.norm(enriched_query))
        else:
            output = enriched_query
        # out: (B, 1, D) -> (B, D), or no ops for more than 1 query
        return torch.squeeze(output), attn

class LearnedQueryAttentionV5(nn.Module, HyperParameters):
    """
    Multihead attention feature aggregation module.

    Inputs:
        - x: Tensor of shape (B, X, Y, D)
    Outputs:
        - aggregated: Tensor of shape (B, D)
    """

    def __init__(self, 
        aggregation_dim,
        dim, nhead,  
        activation, use_ffn=False, dropout=0.1,
        num_queries=1
    ):
        super(LearnedQueryAttentionV5, self).__init__()
        self.save_hyperparameters()
        assert self.aggregation_dim == [1, 2] or self.aggregation_dim == [1], "aggregation_dim must be either [1, 2] or [1]"
        
        self.query_embed = nn.Embedding(self.num_queries, self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.nhead, batch_first=True)
        self.dropout_after_attn = nn.Dropout(self.dropout)
        
        if self.use_ffn:
            self.norm = nn.LayerNorm(self.dim)
            self.ffn = nn.Sequential(
                nn.Linear(self.dim, self.dim * 4),
                get_activation(activation=self.activation, functional=False)(),
                nn.Dropout(self.dropout),
                nn.Linear(self.dim * 4, self.dim),
                nn.Dropout(self.dropout)
            ) 

    def forward(self, x):
        if self.aggregation_dim == [1, 2]:
            assert len(x.shape) == 4, "Expected x to be of shape (B, X, Y, D)"
            B, X, Y, D = x.shape
            x = x.reshape(B, X * Y, D)
        elif self.aggregation_dim == [1]:
            assert len(x.shape) == 3, "Expected x to be of shape (B, X, D)"
            B, X, D = x.shape
        else:
            raise ValueError("aggregation_dim should be either [1, 2] or [1]")
        assert D == self.dim, "Expected last dim to be of length {} but got {}".format(self.dim, D)

        input_query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        extracted, attn = self.attn(input_query, x, x)
        enriched_query = self.dropout_after_attn(extracted)

        if self.use_ffn:
            output = self.ffn(self.norm(enriched_query))
        else:
            output = enriched_query
        # out: (B, 1, D) -> (B, D), or no ops for more than 1 query
        return torch.squeeze(output), attn

# V0 no norms and no residual without final dropout
# V1 no norms and no residual
# V2 2 norms and residual
# V3 no norms and residual
# V4 1 norm and residual
# V5 1 norm and no residual

class FeatureMapCompressor(nn.Module, HyperParameters):

    def __init__(
        self, 
        # groups of attentions
        num_groups, num_queries_per_group,
        # parameters for LearnedQueryAttentionFeatureAggregators
        dim, nhead, activation, 
        use_ffn=True, dropout=0.1, use_norm=False, norm_first=False,
    ):
        super(FeatureMapCompressor, self).__init__()
        self.save_hyperparameters()
        self.attention_list = nn.ModuleList([
            LearnedQueryAttentionFeatureAggregator(
                dim=self.dim, nhead=self.nhead, aggregation_dim=[1, 2], 
                activation=self.activation, use_ffn=self.use_ffn, dropout=self.dropout,
                use_norm=self.use_norm, norm_first=self.norm_first,
                num_queries=self.num_queries_per_group
            ) for _ in range(self.num_groups)
        ])

    def forward(self, x):
        # x: (B * W, C, H, W)
        # output: (B * W, num_queries, dim)
        x = x.permute(0, 2, 3, 1)
        result_list = []
        for attention in self.attention_list:
            result_list.append(attention(x)[0])
        return torch.cat(result_list, dim=1)


if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)

    print("--- 测试 LearnedQueryAttentionFeatureAggregator ---")
    
    # 1. 测试 [1, 2] 聚合模式 (输入是 4D 张量: B, X, Y, D)
    print("\n1. 测试聚合模式 [1, 2] (输入4D张量):")
    batch_size, height, width, dim = 2, 3, 5, 256
    nhead = 8
    num_queries = 7
    
    # 创建模块实例
    aggregator_4d = LearnedQueryAttentionV2(
        dim=dim, 
        nhead=nhead, 
        aggregation_dim=[1, 2],
        activation='relu',
        use_ffn=True,
        dropout=0.1,
        # use_norm=True,
        norm_first=False,
        num_queries=num_queries
    )
    
    # 创建随机输入
    x_4d = torch.randn(batch_size, height, width, dim)
    
    # 前向传播
    output_4d, attn_weights_4d = aggregator_4d(x_4d)
    
    print(f"  输入形状: {x_4d.shape}")
    print(f"  输出特征形状: {output_4d.shape}")
    print(f"  注意力权重形状: {attn_weights_4d.shape}")
    # 预期输出形状: (B, num_queries * dim) 当 num_queries > 1 时
    # 或 (B, dim) 当 num_queries == 1 时
    expected_output_shape_4d = (batch_size, num_queries, dim) if num_queries > 1 else (batch_size, dim)
    print(f"  预期输出形状: {expected_output_shape_4d}")
    assert output_4d.shape == expected_output_shape_4d, "4D模式下输出形状不正确！"
    print("  [1, 2] 聚合模式测试通过！")

    # 2. 测试 [1] 聚合模式 (输入是 3D 张量: B, X, D)
    print("\n2. 测试聚合模式 [1] (输入3D张量):")
    batch_size, sequence_length, dim = 2, 7, 256
    
    # 创建模块实例
    aggregator_3d = LearnedQueryAttentionV2(
        dim=dim, 
        nhead=nhead, 
        aggregation_dim=[1],
        activation='relu',
        use_ffn=False, # 这次不使用FFN
        num_queries=1 # 只使用1个query
    )
    
    # 创建随机输入
    x_3d = torch.randn(batch_size, sequence_length, dim)
    
    # 前向传播
    output_3d, attn_weights_3d = aggregator_3d(x_3d)
    
    print(f"  输入形状: {x_3d.shape}")
    print(f"  输出特征形状: {output_3d.shape}")
    print(f"  注意力权重形状: {attn_weights_3d.shape}")
    expected_output_shape_3d = (batch_size, dim)
    print(f"  预期输出形状: {expected_output_shape_3d}")
    assert output_3d.shape == expected_output_shape_3d, "3D模式下输出形状不正确！"
    print("  [1] 聚合模式测试通过！")

    print("\n--- 测试 FeatureMapCompressor ---")
    
    # 准备输入 (模拟CNN输出的特征图)
    batch_size, channels, height, width = 2, 256, 5, 7
    
    # 创建 Compressor 实例
    compressor = FeatureMapCompressor(
        num_groups=3,
        num_queries_per_group=7,
        dim=channels, # 输入通道数应与 dim 匹配
        nhead=8,
        activation='relu',
        use_ffn=True,
        use_norm=True
    )
    
    # 创建随机输入特征图
    x_feature_map = torch.randn(batch_size, channels, height, width)
    
    # 前向传播
    compressed_features = compressor(x_feature_map)
    
    print(f"\n输入特征图形状: {x_feature_map.shape}")
    print(f"压缩后特征形状: {compressed_features.shape}")
    # 预期输出形状: (num_groups * num_queries_per_group, B, dim)
    expected_compressed_shape = (batch_size, compressor.num_groups * compressor.num_queries_per_group, channels)
    print(f"预期压缩后形状: {expected_compressed_shape}")
    assert compressed_features.shape == expected_compressed_shape, "FeatureMapCompressor 输出形状不正确！"
    print("FeatureMapCompressor 测试通过！")

    print("\n所有测试均已通过！")
