import torch
import torch.nn as nn
import torch.nn.functional as F
from .zim_cnn import build_zim_cnn
from .zim_position_embedding import build_position_embedding
from .zim_window_encoder import WindowEncoder
from .zim_channel_attention import ChannelAttention
from util.utils import get_activation

class ZimBackbone(nn.Module):
    def __init__(self, args):
        super(ZimBackbone, self).__init__()
        
        self.cnn = build_zim_cnn(args)
        self.position_embedding = build_position_embedding(args)

        self.num_channels_for_each_layer_of_resnet18 = [-1, 64, 128, 256, 512]
        adaptors = []
        for layer_index in args.scale_selection_from_cnn:
            adaptors.append(
                ChannelAlign(
                    self.num_channels_for_each_layer_of_resnet18[layer_index], 
                    args.hidden_dim,
                    use_bn=args.use_bn_for_adaptors,
                    activation=args.activation_for_all_of_zim_head
                )
                if not args.use_channel_attention_in_adaptors else
                nn.Sequential(
                    ChannelAttention(
                        in_channels=self.num_channels_for_each_layer_of_resnet18[layer_index],
                        activation=args.activation_for_all_of_zim_head,
                        residual_connection=args.residual_connection_in_channel_attentions
                    ),
                    ChannelAlign(
                        self.num_channels_for_each_layer_of_resnet18[layer_index], 
                        args.hidden_dim,
                        use_bn=args.use_bn_for_adaptors,
                        activation=args.activation_for_all_of_zim_head
                    )
                )
            )
        self.adaptors = nn.ModuleList(adaptors)
        
        window_encoders = []
        for _ in args.scale_selection_from_cnn:
            window_encoders.append(WindowEncoder(
                # tokens
                num_tokens=args.num_tokens, hidden_dim=args.hidden_dim,
                # transformer
                nhead=args.token_encoder_nhead, normalize_before=args.token_encoder_norm_first, 
                dim_feedforward=args.token_encoder_dim_feedforward, dropout=args.token_encoder_dropout, activation=args.activation_for_all_of_zim_head, 
                num_decoder_layers=args.token_encoder_nlayers, return_intermediate_dec=args.token_encoder_return_intermediate,
                # st enhancer
                num_time_enc_layers=args.west_num_time_enc_layers, 
                num_space_enc_layers=args.west_num_space_enc_layers,
                num_time_dec_layers=args.west_num_time_dec_layers,
                num_space_dec_layers=args.west_num_space_dec_layers,
                use_time_positional=args.use_time_positional,
                max_time=args.window_width,
                use_space_positional=args.use_space_positional,
                max_space=args.num_tokens,
                # pooling method
                pooling_method = args.pooling_method, 
                moe_num_experts=args.moe_num_experts,
                mean_residual_connection_for_pooling=args.mean_residual_connection_for_pooling,
                use_ffn_in_aggregation=args.use_ffn_in_aggregation,
                noise_gating=args.use_noise_gating_in_moe
            ))
        self.window_encoders = nn.ModuleList(window_encoders)
        
    def forward(self, x):
        # x: [batch_size * num_windows, window_width, C, H, W]
        # output: [batch_size * num_windows, num_scales, hidden_dim]
        
        # pass through CNN
        bs_x_nw, window_width, C, H, W = x.shape
        x = x.reshape(bs_x_nw * window_width, C, H, W)
        feature_map_list = self.cnn(x)
        
        # get position embeddings for maps
        position_embedding_list = []
        for f in feature_map_list:
            position_embedding_list.append(self.position_embedding(f).to(f.dtype))
        
        # adapt the channel dimension to hidden_dim for feature maps
        for index, feature_of_some_scale in enumerate(feature_map_list):
            feature_map_list[index] = self.adaptors[index](feature_of_some_scale)

        # Now feature_map_list and position_embedding_list are of size
        # list[torch.Size([bs * nw * ww, hidden_dim, x, y]), ..., ..., ...]
        
        
        for index, _ in enumerate(feature_map_list):
            feature_map_list[index] = feature_map_list[index].reshape(
                bs_x_nw, window_width, 
                -1, feature_map_list[index].shape[-2], feature_map_list[index].shape[-1]
            )
            position_embedding_list[index] = position_embedding_list[index].reshape(
                bs_x_nw, window_width,
                -1, position_embedding_list[index].shape[-2], position_embedding_list[index].shape[-1]
            )
        
        
        for index, feature in enumerate(feature_map_list):
            feature_map_list[index] = self.window_encoders[index](
                feature, position_embedding_list[index]
            )
        
        return torch.stack(feature_map_list, dim=0).permute(1, 0, 2)


class ChannelAlign(nn.Module):
    """
    使用线性层将特征图的通道数从 C 对齐到 D。
    输入:  (B, C, H, W)
    输出:  (B, D, H, W)
    """
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = False, activation: str = 'none'):
        super(ChannelAlign, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False if use_bn else True)
        self.use_bn = use_bn
        self.activation = activation
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation != 'none':
            self.activation_fn = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 交换通道维到最后，以便Linear作用在C维
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.linear(x)         # (B, H, W, D)
        x = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        if self.use_bn:
            x = self.bn(x)
        if self.activation != 'none':
            x = self.activation_fn(x)

        return x

