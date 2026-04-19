# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/transformer.py
# ------------------------------------------------------------------------
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from util.utils import HyperParameters, get_activation

class Transformer(nn.Module, HyperParameters):

    def __init__(
            self, 
            d_model, nhead, normalize_before,
            dim_feedforward, dropout, activation, 
            return_intermediate_dec, num_decoder_layers
        ):
        super().__init__()
        self.save_hyperparameters()
        # d_model: the number of expected features in the input and output embeddings
        # nhead: the number of heads in the multiheadattention models
        # num_decoder_layers: the number of sub-decoder-layers in the decoder
        # dim_feedforward: the hidden intermediate dimension of the feedforward network model
        # dropout: the dropout value for the feedforward network
        # activation: the activation function of encoder/decoder intermediate layer, relu or gelu
        # normalize_before: whether to use layer_norm before the first decoder layer
        # return_intermediate_dec: whether to return the output of intermediate decoder layers
        # original defaults from the DFWSGAR codebase:  
        # d_model=512, nhead=8, 
        # num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
        # activation="relu", normalize_before=False,
        # return_intermediate_dec=False
        # DFWSGAR deletes the encoders in the original facebook codebase

        decoder_layer = TransformerDecoderLayer(
                            d_model, nhead, dim_feedforward,
                            dropout, activation, normalize_before
                        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
                            decoder_layer, num_decoder_layers, 
                            decoder_norm, return_intermediate_dec
                        )
        # self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # all inputs should be of shape [seq_len, batch_size, d_model]
        # src - keys and values, query_embed - queries, mask and pos_embed are for src
        # output - shape [seq_len, batch_size, d_model]
        tgt = torch.zeros_like(query_embed)
        return self.decoder(tgt, src, memory_key_padding_mask=mask,
                               pos=pos_embed, query_pos=query_embed)


class TransformerDecoder(nn.Module, HyperParameters):

    def __init__(
        self, 
        decoder_layer, num_layers, 
        norm, return_intermediate
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['decoder_layer'])  # decoder_layer is just for cloning
        
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        self.layers = _get_clones(decoder_layer, num_layers)


    def forward(
        self, 
        tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ):
        output = tgt
        att = None

        intermediate = []
        intermediate_att = []

        for layer in self.layers:
            output, att = layer(output, memory, tgt_mask=tgt_mask,
                                memory_mask=memory_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask,
                                pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_att.append(att)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_att)

        return output, att

class TransformerDecoderLayer(nn.Module, HyperParameters):

    def __init__(
        self, 
        d_model, nhead, 
        dim_feedforward, dropout,
        activation, normalize_before
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt2, att = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, att

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        tgt2, att = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                        key=self.with_pos_embed(memory, pos),
                                        value=memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt, att

    def forward(
        self,
        tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
