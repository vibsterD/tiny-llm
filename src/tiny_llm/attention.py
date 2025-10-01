import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    if scale is None:
        scale = 1 / mx.sqrt(query.shape[-1])

    attn_scores = query @ key.swapaxes(-2, -1)
    attn_scores = attn_scores * scale
    if mask is not None:
        attn_scores = attn_scores + mask
    attn_scores = mx.softmax(attn_scores, axis=-1)
    return attn_scores @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.num_heads = num_heads
        self.hidden_size = hidden_size

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:

        query = query @ self.wq.swapaxes(-1, -2)
        key = key @ self.wk.swapaxes(-1, -2)
        value = value @ self.wv.swapaxes(-1, -2)

        # (N, L, HxD) -> (N, H, L, D)
        query = query.reshape(query.shape[0], query.shape[1], self.num_heads, -1).swapaxes(-2, -3)
        key = key.reshape(key.shape[0], key.shape[1], self.num_heads, -1).swapaxes(-2, -3)
        value = value.reshape(value.shape[0], value.shape[1], self.num_heads, -1).swapaxes(-2, -3)
        attn_scores = scaled_dot_product_attention_simple(query, key, value, mask=mask)

        # (N, H, L, D) -> (N, L, HxD)
        attn_scores = attn_scores.swapaxes(-2, -3).reshape(query.shape[0], query.shape[2], -1)
        return attn_scores @ self.wo.swapaxes(-1, -2)



def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    # query: N.. x H_q x L x D
    # key: N.. x H x S x D
    # value: N.. x H x S x D
    # mask: N.. x H_q x L x S
    # output: N.. x H_q x L x D

    if scale is None:
        scale = 1 / mx.sqrt(query.shape[-1])

    B = query.shape[:-3]
    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    
    assert H_q % H == 0

    n_repeats = H_q // H

    # How to utilise broadcasting? 
    # query: B x H_q x L x D -> B x H x n_repeats x L x D
    # key: B x H x S x D -> B x H x 1 x S x D 
    # value: B x H x S x D -> B x H x 1 x S x D
    # mask: B x H_q x L x S -> B x H x n_repeats x L x S

    query = query.reshape(-1, H, n_repeats, L, D)
    key = key.reshape(-1, H, 1, S, D)
    value = value.reshape(-1, H, 1, S, D)
    
    attn_scores = query @ key.swapaxes(-2, -1)
    attn_scores = attn_scores * scale

    if mask is not None:
        mask = mask.reshape(-1, H, n_repeats, L, S)
        attn_scores = attn_scores + mask
    
    attn_scores = softmax(attn_scores, axis=-1)
    attn_scores = attn_scores @ value
    return attn_scores.reshape(*B, H_q, L, D)



def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
