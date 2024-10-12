import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .arm import AttentionRefinementModule
from torch import Tensor
from einops import rearrange

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

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out, bias=False)

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
                            B is the batch size, and D is the embedding dimension.
            key (Tensor): The key embeddings of shape (B, L, D), where S is the source sequence length,
                          B is the batch size, and D is the embedding dimension.
            value (Tensor): The value embeddings of shape (B, L, D), where S is the source sequence length,
                            B is the batch size, and D is the embedding dimension.
            key_padding_mask (Optional[Tensor]): If provided, specifies padding elements in the key.
                                                 Shape should be (B, S), where B is the batch size and S is the source sequence length.
            attn_mask (Optional[Tensor]): If provided, specifies positions that should be masked.
                                          Shape should be (L, S), where L is the target sequence length and S is the source sequence length.
            target_vocab (Optional[Tensor]): If provided, specifies the target vocabulary.
                                            Shape should be (B, L), where B is the batch size and L is the target sequence length.
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
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # Shape: (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, -1, self.num_kv_groups, self.head_dim) # Shape: (b, num_tokens, num_kv_groups, head_dim)
        values = values.view(b, -1, self.num_kv_groups, self.head_dim) # Shape: (b, num_tokens, num_kv_groups, head_dim)

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Expand keys and values to match the number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
         # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]
        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
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

        # Apply softmax and dropout
        attn_output_weights = F.softmax(attn_scores, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        # if arm is not None and target_vocab is not None:
        #     attention_refine = arm(curr_attn = rearrange(attn_scores, "b n t s -> (b n) t s"), 
        #                            tgt_vocab = target_vocab)
        #     attention_refine_reshape = rearrange(attention_refine, "(b n) t s -> b n t s", b=b)
        #     attn_output_weights -= attention_refine_reshape

        context_vec = (attn_output_weights @ values).transpose(1, 2) # Shape: (b, num_tokens, num_heads, head_dim)

        # context_vec = context_vec.transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        attn_output = self.out_proj(context_vec)

        attn_output_weights = rearrange(attn_output_weights, "b n t s -> (b n) t s")

        if need_weights:
            return attn_output, attn_output_weights
        else:
            return attn_output
