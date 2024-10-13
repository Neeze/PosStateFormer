import copy
from functools import partial
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .arm import AttentionRefinementModule
from .attention import MultiheadAttention
from .group_query_attention import GroupedQueryAttention
from einops import rearrange


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        arm: Optional[AttentionRefinementModule],
        norm=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.arm = arm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        height: int,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_vocab:Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt

        arm = None
        for i, mod in enumerate(self.layers):
            output, attn = mod(
                output,
                memory,
                arm,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_vocab=tgt_vocab
            )
            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(self.arm, attn, memory_key_padding_mask, height)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn


# class FeedForward(nn.Module):
#     def __init__(self, d_model, dim_feedforward):
#         super().__init__()
#         self.fc1 = nn.Linear(d_model, dim_feedforward, bias=False)
#         self.fc2 = nn.Linear(d_model, dim_feedforward, bias=False)
#         self.fc3 = nn.Linear(dim_feedforward, d_model, bias=False)

#     def forward(self, x):
#         x_fc1 = self.fc1(x)
#         x_fc2 = self.fc2(x)
#         x = nn.functional.silu(x_fc1) * x_fc2
#         return self.fc3(x)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, d_model, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc1 = nn.functional.silu(x_fc1)
        x_fc1 = self.dropout(x_fc1)
        return self.fc2(x_fc1)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_kv_groups, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead // 2, dropout=dropout)
        # self.self_attn = GroupedQueryAttention(d_in=d_model,
        #                                         d_out=d_model,
        #                                         num_heads=nhead,
        #                                         num_kv_groups=num_kv_groups,
        #                                         dropout=dropout)
        # self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.group_attn = GroupedQueryAttention(d_in=d_model,
                                                d_out=d_model,
                                                num_heads=nhead,
                                                num_kv_groups=num_kv_groups,
                                                dropout=dropout)

        self.ff = FeedForward(d_model=d_model, 
                              dim_feedforward=dim_feedforward, 
                              dropout=dropout)

        self.norm1 = nn.RMSNorm(d_model, eps=1e-5)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-5)
        self.norm3 = nn.RMSNorm(d_model, eps=1e-5)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # self.activation = F.relu

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        arm: Optional[AttentionRefinementModule],
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_vocab:Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt = rearrange(tgt, "b l d -> l b d")

        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = rearrange(tgt, "l b d -> b l d")

        # Implement Group Query Attention
        tgt2, attn = self.group_attn(
            query=tgt,
            key=memory,
            value=memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            target_vocab=tgt_vocab,
        )

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt2 = self.ff(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn
