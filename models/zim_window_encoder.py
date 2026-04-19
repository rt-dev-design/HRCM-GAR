import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.utils import HyperParameters, get_activation
from .zim_transformer import Transformer
from .zim_st_enhancer import SpatialTemporalEnhancer
from .zim_aggregators import *
from .zim_moe import MoeAttentionAggregator

class WindowEncoder(nn.Module, HyperParameters):
    def __init__(
        self, 
        # tokens
        num_tokens, hidden_dim,
        # transformer
        nhead, normalize_before, 
        dim_feedforward, dropout, activation, 
        num_decoder_layers, return_intermediate_dec,
        # st enhancer
        num_time_enc_layers, 
        num_space_enc_layers, 
        num_time_dec_layers,
        num_space_dec_layers,
        use_time_positional,
        max_time,
        use_space_positional,
        max_space,
        # pooling
        pooling_method, moe_num_experts,
        mean_residual_connection_for_pooling,
        use_ffn_in_aggregation,
        noise_gating
    ):
        super(WindowEncoder, self).__init__()
        self.save_hyperparameters()
        
        self.embedding_to_implement_tokens = nn.Embedding(self.num_tokens, self.hidden_dim)

        self.token_encoder = Transformer(
            d_model=self.hidden_dim, nhead=self.nhead, normalize_before=self.normalize_before,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation, 
            num_decoder_layers=self.num_decoder_layers, return_intermediate_dec=self.return_intermediate_dec
        )

        self.st_enhancer = SpatialTemporalEnhancer(
            d_model=self.hidden_dim, nhead=self.nhead, norm_first=self.normalize_before,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout, activation=self.activation,
            num_time_enc_layers=self.num_time_enc_layers,
            num_space_enc_layers=self.num_space_enc_layers,
            num_time_dec_layers=self.num_time_dec_layers,
            num_space_dec_layers=self.num_space_dec_layers,
            use_time_positional=self.use_time_positional,
            max_time=self.max_time,
            use_space_positional=self.use_space_positional,
            max_space=self.max_space
        )

        self.pooling = None
        if self.pooling_method == 'mean':
            self.pooling = lambda t: torch.mean(input=t, dim=[1, 2])
        elif self.pooling_method == 'max':
            self.pooling = lambda t: torch.amax(input=t, dim=[1, 2])
        elif self.pooling_method == 'attn':
            self.pooling = LearnedQueryAttentionV1(
                aggregation_dim=[1, 2],
                dim=self.hidden_dim, nhead=self.nhead,
                activation=self.activation,
                use_ffn=self.use_ffn_in_aggregation
            )
        elif self.pooling_method == 'mean_max':
            self.pooling = lambda t: torch.mean(input=t, dim=[1, 2]) + torch.amax(input=t, dim=[1, 2])
        elif self.pooling_method == 'moe':
            self.pooling = MoeAttentionAggregator(
                hidden_dim=self.hidden_dim, 
                experts=[LearnedQueryAttentionV1(
                    aggregation_dim=[1, 2],
                    dim=self.hidden_dim, nhead=self.nhead,
                    activation=self.activation,
                    use_ffn=self.use_ffn_in_aggregation
                ) for _ in range(self.moe_num_experts)],
                k=self.moe_num_experts,
                noisy_gating=self.noise_gating
            )
        else:
            assert False, "Unsupported pooling method: {}. Choose between mean, max, attn, or mean_max.".format(self.pooling_method)

    def forward(self, x, pos):
        # x: bs * nw, ww, hidden_dim, h, w
        # The input can be viewed as a bag of windows
        # pos: of the same shape as x
        # output: bs * nw, hidden_dim
        
        bs_x_nw, window_width, hidden_dim, h, w = x.shape
        batch_for_transformer = bs_x_nw * window_width

        # x and pos are supposed to be of shape (seq_len, batch, dim) for Transformer
        # flatten NxCxHxW to HWxNxC
        x = x.reshape(batch_for_transformer, -1, h, w).flatten(2).permute(2, 0, 1)
        pos = pos.reshape(batch_for_transformer, -1, h, w).flatten(2).permute(2, 0, 1)
        tokens = self.embedding_to_implement_tokens.weight.unsqueeze(1).repeat(1, batch_for_transformer, 1)
        encoded_tokens, att = self.token_encoder(x, None, tokens, pos)
        # encoded_tokens, (seq_len, batch_for_transformer, dim), (num_tokens, bs_x_nw * window_width, dim)
        
        # temporal and token aggregation
        representations = encoded_tokens.reshape(self.num_tokens, bs_x_nw, window_width, self.hidden_dim).permute(1, 2, 0, 3)
        
        representations = self.st_enhancer(representations)

        if not self.mean_residual_connection_for_pooling:
            representations = self.pooling(representations) if self.pooling_method != 'attn' else self.pooling(representations)[0]
        else:
            representations_mean = torch.mean(representations, dim=[1, 2])
            representations = representations_mean + (self.pooling(representations) if self.pooling_method != 'attn' else self.pooling(representations)[0])
        
        return representations

