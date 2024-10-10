import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .arm import AttentionRefinementModule
from torch import Tensor
from einops import rearrange

def precompute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)

        if key not in SharedBuffers._buffers:
            # Create or fetch the buffers
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(head_dim, rope_base, context_length, freq_config)
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)
            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]

class GroupedQueryAttention(nn.Module):
    def __init__(
            self, 
            d_in: int, 
            d_out: int,
            num_heads: int,
            num_kv_groups: int,
            dropout: float = 0.0,
        ):
        super(GroupedQueryAttention, self).__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=True)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=True)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=True)
        self.out_proj = nn.Linear(d_out, d_out, bias=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                query: Tensor, 
                key: Tensor, 
                value: Tensor,
                arm: Optional[AttentionRefinementModule] = None, 
                key_padding_mask: Optional[Tensor] = None, 
                need_weights: bool = True,
                attn_mask: Optional[Tensor] = None,
                target_vocab: Optional[Tensor] = None):
        """
        Forward pass for GroupedQueryAttention module.
        
        Args:
            query (Tensor): The query embeddings of shape (B, L, D), where L is the target sequence length,
                            N is the batch size, and E is the embedding dimension.
            key (Tensor): The key embeddings of shape (B, L, D), where S is the source sequence length,
                          N is the batch size, and E is the embedding dimension.
            value (Tensor): The value embeddings of shape (B, L, D), where S is the source sequence length,
                            N is the batch size, and E is the embedding dimension.
            key_padding_mask (Optional[Tensor]): If provided, specifies padding elements in the key.
                                                 Shape should be (B, L), where N is the batch size and S is the source sequence length.
            attn_mask (Optional[Tensor]): If provided, specifies positions that should be masked.
                                          Shape should be (L, S), where L is the target sequence length and S is the source sequence length.
            target_vocab (Optional[Tensor]): If provided, specifies the target vocabulary.
                                            Shape should be (B, L), where N is the batch size and L is the target sequence length.
        Returns:
            attn_output (Tensor): The output embeddings of shape (B, L, D), where L is the target sequence length,
                                  N is the batch size, and E is the embedding dimension.
            attn_output_weights (Optional[Tensor]): The attention weights of shape (B, L, S), where N is the batch size,
                                                    L is the target sequence length, and S is the source sequence length.
        """
        
        b, num_tokens, d_in = query.shape

        # Project query, key, and value
        queries = self.W_query(query)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(key)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(value)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, -1, self.num_kv_groups, self.head_dim)
        values = values.view(b, -1, self.num_kv_groups, self.head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Expand keys and values to match the number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)

        # Compute scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Handle key_padding_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        # Handle attn_mask
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 1, float('-inf'))

        attn_output_weights = attn_scores
        # Apply ARM if provided
        # if arm is not None and target_vocab is not None:
        #     attention_refine = arm(rearrange(attn_scores, "b n t s -> (b n) t s"), target_vocab)
        #     attention_refine_reshape = rearrange(attention_refine, "(b n) t s -> b n t s", b=b)
        #     attn_output_weights -= attention_refine_reshape
        # else:
        #     attn_output_weights = attn_scores

        # Apply softmax and dropout
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        context_vec = (attn_output_weights @ values).transpose(1, 2)

        # Transpose context_vec to match the original shape
        context_vec = context_vec.transpose(1, 2)  # (b, num_tokens, num_query_groups, head_dim)

        # Combine heads        
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)

        # Final linear projection
        attn_output = self.out_proj(context_vec)

        attn_output_weights = rearrange(attn_output_weights, "b n t s -> (b n) t s")

        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output
