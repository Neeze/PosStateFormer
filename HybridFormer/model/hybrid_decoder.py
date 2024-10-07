import copy
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor
from einops import rearrange

from mambapy.mamba import MambaConfig, MambaBlock, RMSNorm
from HybridFormer.model.transformer.transformer_decoder import (
    AttentionRefinementModule,
    MultiheadAttention,
)

from HybridFormer.datamodule import vocab, vocab_size 
from HybridFormer.model.pos_enc import WordPosEnc
from HybridFormer.utils.generation_utils import DecodeModel, PosDecodeModel


class BiMambaBlock(nn.Module):
    def __init__(self, d_model, n_state):
        super(BiMambaBlock, self).__init__()
        self.d_model = d_model
        self.mamba_config = MambaConfig(n_layers=1, 
                                        d_model=d_model, 
                                        d_state=n_state, 
                                        bias=True)
        self.mamba = MambaBlock(self.mamba_config)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # Combining forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        output = ff_out + residual
        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class HybridDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers: int,
        arm: Optional[AttentionRefinementModule],
        norm=None,
    ):
        super(HybridDecoder, self).__init__()
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


class HybridDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nhead, 
                 d_state: int, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 eps=1e-5,):
        super(HybridDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Mamba model
        self.mamba = BiMambaBlock(d_model, d_state)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        # self.norm4 = nn.LayerNorm(d_model)

        # Implementation of RMSNorm
        self.norm1 = RMSNorm(d_model=d_model, eps=eps)
        self.norm2 = RMSNorm(d_model=d_model, eps=eps)
        self.norm3 = RMSNorm(d_model=d_model, eps=eps)
        self.norm4 = RMSNorm(d_model=d_model, eps=eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(HybridDecoderLayer, self).__setstate__(state)

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
                [l, b, d]
            memory: the sequence from the last layer of the encoder (required).
                [l, b, d]
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
                [b, l]

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            tgt_vocab=tgt_vocab,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt) # l b d 

        tgt = rearrange(tgt, "l b d -> b l d")
        tgt = tgt + self.dropout3(self.mamba(tgt))
        tgt = rearrange(tgt, "b l d -> l b d")
        tgt = self.norm3(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, attn


def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    d_state: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
    dc: int,
    cross_coverage: bool,
    self_coverage: bool,
) -> HybridDecoder:
    decoder_layer = HybridDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        d_state=d_state, 
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    if cross_coverage or self_coverage:
        arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)
    else:
        arm = None

    decoder = HybridDecoder(decoder_layer, num_decoder_layers, arm)
    return decoder

class Decoder(DecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_state: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )
        self.pos_enc = WordPosEnc(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            d_state=d_state,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor 
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]
        pos_labels:LongTensor
            [b,l,5]
        is_test:bool
        
        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt_vocab=tgt
        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]
        tgt = self.norm(tgt)
        
        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out, attn  = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
        )
    
        out_rearrange = rearrange(out, "l b d -> b l d")
        out = self.proj(out_rearrange)
        return out, attn

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        word_out, _ = self(src[0], src_mask[0], input_ids)
        return word_out


class PosDecoder(PosDecodeModel):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_state: int,   
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        self.pos_embed = nn.Sequential(
            nn.Linear(5,d_model),nn.GELU(),nn.LayerNorm(d_model)
        )  #[2b,l,5]  -->  [2b,l,256]
        self.pos_enc = WordPosEnc(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            d_state=d_state,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )
        self.layernum_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        ) 
        self.pos_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        ) 
    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal    

        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor,pos_tgt:FloatTensor
    ) -> Tuple[ FloatTensor,FloatTensor]:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]
        pos_labels:LongTensor
            [b,l,5]
        is_test:bool
        
        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """

        b , l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX
        tgt_vocab=tgt
        pos_tgt=self.pos_embed(pos_tgt)  #[b,l,d]  
        pos_tgt = self.pos_enc(pos_tgt)  # [b, l, d]
        pos_tgt = self.norm(pos_tgt)


        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        pos_tgt = rearrange(pos_tgt, "b l d -> l b d")

        out, attn = self.model(
            tgt=pos_tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
            tgt_vocab=tgt_vocab,
        )
        out_rearrange = rearrange(out, "l b d -> b l d")
        out_pos=self.pos_proj(out_rearrange)
        out_layernum=self.layernum_proj(out_rearrange)
        return out_layernum , out_pos, attn

    def transform(
        self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        out_pos, _ = self(src[0], src_mask[0], input_ids,torch.zeros(1, dtype=torch.float, device=self.device))
        return out_pos
