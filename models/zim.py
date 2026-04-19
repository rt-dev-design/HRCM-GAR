import torch
import torch.nn as nn
import torch.nn.functional as F
from .zim_backbone import ZimBackbone
from util.utils import HyperParameters
from .zim_st_enhancer import SpatialTemporalEnhancer
from .zim_aggregators import *
from .zim_moe import MoeAttentionAggregator

class ZimBasic(nn.Module, HyperParameters):
    def __init__(self, args):
        super(ZimBasic, self).__init__()
        self.save_hyperparameters()
        self.zim_backbone = ZimBackbone(args)
        if args.use_clip_scale_st:
            self.window_scale_enhancer = SpatialTemporalEnhancer(
                d_model=args.hidden_dim, nhead=args.token_encoder_nhead, norm_first=args.token_encoder_norm_first,
                dim_feedforward=args.token_encoder_dim_feedforward, dropout=args.dropout, activation=args.activation_for_all_of_zim_head,
                num_time_enc_layers=args.csst_num_time_enc_layers,
                num_space_enc_layers=args.csst_num_space_enc_layers,
                num_time_dec_layers=args.csst_num_time_dec_layers,
                num_space_dec_layers=args.csst_num_space_dec_layers,
                use_time_positional=args.use_time_positional,
                max_time=args.num_windows,
                use_space_positional=args.use_space_positional,
                max_space=args.num_scales
            )
        self.pooling = None
        if args.pooling_method == 'mean':
            self.pooling = lambda t: torch.mean(input=t, dim=[1, 2])
        elif args.pooling_method == 'max':
            self.pooling = lambda t: torch.amax(input=t, dim=[1, 2])
        elif args.pooling_method == 'attn':
            self.pooling = LearnedQueryAttentionV1(
                aggregation_dim=[1, 2],
                dim=args.hidden_dim, nhead=args.token_encoder_nhead,
                activation=args.activation_for_all_of_zim_head,
                use_ffn=args.use_ffn_in_aggregation
            )
        elif args.pooling_method == 'mean_max':
            self.pooling = lambda t: torch.mean(input=t, dim=[1, 2]) + torch.amax(input=t, dim=[1, 2])
        elif args.pooling_method == 'moe':
            self.pooling = MoeAttentionAggregator(
                hidden_dim=args.hidden_dim, 
                experts=[LearnedQueryAttentionV1(
                    aggregation_dim=[1, 2], 
                    dim=args.hidden_dim, nhead=args.token_encoder_nhead,
                    activation=args.activation_for_all_of_zim_head,
                    use_ffn=args.use_ffn_in_aggregation
                ) for _ in range(args.moe_num_experts)],
                k=args.moe_num_experts,
                noisy_gating=args.use_noise_gating_in_moe
            )
        else:
            assert False, "Unsupported pooling method: {}. Choose between mean, max, attn, or mean_max.".format(args.pooling_method)
        self.classifier = nn.Linear(args.hidden_dim, args.num_classes)
    
    def forward(self, x):
        # x: batch_size, num_windows, window_width, 3, H, W
        batch_size, num_windows, window_width, C, H, W = x.shape
        x = x.reshape(batch_size * num_windows, window_width, C, H, W)
        x = self.zim_backbone(x)
        x = x.reshape(batch_size, num_windows, self.args.num_scales, self.args.hidden_dim)
        if self.args.use_clip_scale_st:
            x = self.window_scale_enhancer(x)
        if not self.args.mean_residual_connection_for_pooling:
            x = self.pooling(x) if self.args.pooling_method != 'attn' else self.pooling(x)[0]
        else:
            x_mean = torch.mean(x, dim=[1, 2])
            x = x_mean + (self.pooling(x) if self.args.pooling_method != 'attn' else self.pooling(x)[0])
        return self.classifier(x)
    
class ZimFull(nn.Module, HyperParameters):
    def __init__(self, args):
        super(ZimFull, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

def build_zim(args):
    if (args.zim_type == "basic"):
        return ZimBasic(args)
    elif (args.zim_type == "full"):
        return ZimFull(args)
    else:
        assert False, "Unsupported zim type: {}. Choose between basic and full.".format(args.zim_type)
