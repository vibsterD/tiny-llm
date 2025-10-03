import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    # x: B, L, E
    # q = linear(x, wq, bq) -> B, L, H_q, D
    # k = linear(x, wk, bk) -> B, L, H, D
    # v = linear(x, wv, bv) -> B, L, H, D
    # q = rope(q, offset=slice(0, L))
    # k = rope(k, offset=slice(0, L))
    # (transpose as needed)
    # x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
    # (transpose as needed)
    # x = linear(x, wo) -> B, L, E

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads # query heads
        self.num_kv_heads = num_kv_heads # key and value heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads} for GQA"
        )
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.rope = RoPE(self.head_dim, max_seq_len, theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape
        
        q_proj = linear(x, self.wq, self.bq).reshape(B, L, self.num_heads, self.head_dim)
        k_proj = linear(x, self.wk, self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v_proj = linear(x, self.wv, self.bv).reshape(B, L, self.num_kv_heads, self.head_dim)

        q_proj = self.rope(q_proj, offset=slice(0, L)).swapaxes(-2, -3).astype(mx.float32)
        k_proj = self.rope(k_proj, offset=slice(0, L)).swapaxes(-2, -3).astype(mx.float32)
        v_proj = v_proj.swapaxes(-2, -3).astype(mx.float32)
        x = scaled_dot_product_attention_grouped(q_proj, k_proj, v_proj, self.scale, mask).swapaxes(-2, -3).reshape(B, L, -1)
        x = linear(x, self.wo).astype(x.dtype)

        return x


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        L, E = x.shape[-2:]
        B = x.shape[:-2]

        up_proj = linear(x, self.w_up)
        gate_proj = silu(linear(x, self.w_gate))
        glu_proj = gate_proj * up_proj
        
        return linear(glu_proj, self.w_down).reshape(*B, L, E)

class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.mha = Qwen2MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x_norm = self.input_layernorm(x)
        x_mha = self.mha(x_norm, mask)
        x_post_attention = x + x_mha
        x_post_attention_norm = self.post_attention_layernorm(x_post_attention)
        x_mlp = self.mlp(x_post_attention_norm)
        return x_post_attention + x_mlp


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):

        self.embedding = Embedding(
            vocab_size=mlx_model.args.vocab_size,
            embedding_dim=mlx_model.args.hidden_size,
            weight=dequantize_linear(mlx_model.model.embed_tokens).astype(mx.float16),
        )

        precision = mx.float16

        self.transformer_blocks = []

        for layer in mlx_model.model.layers:
            q_proj = dequantize_linear(layer.self_attn.q_proj).astype(precision)
            k_proj = dequantize_linear(layer.self_attn.k_proj).astype(precision)
            v_proj = dequantize_linear(layer.self_attn.v_proj).astype(precision)
            wo_proj = dequantize_linear(layer.self_attn.o_proj).astype(precision)
            w_gate = dequantize_linear(layer.mlp.gate_proj).astype(precision)
            w_up = dequantize_linear(layer.mlp.up_proj).astype(precision)
            w_down = dequantize_linear(layer.mlp.down_proj).astype(precision)
            w_input_layernorm = layer.input_layernorm.weight.astype(precision)
            w_post_attention_layernorm = layer.post_attention_layernorm.weight.astype(precision)

            self.transformer_blocks.append( Qwen2TransformerBlock(
                num_attention_heads=mlx_model.args.num_attention_heads,
                num_kv_heads=mlx_model.args.num_key_value_heads,
                hidden_size=mlx_model.args.hidden_size,
                intermediate_size=mlx_model.args.intermediate_size,
                rms_norm_eps=mlx_model.args.rms_norm_eps,
                wq=q_proj,
                wk=k_proj,
                wv=v_proj,
                wo=wo_proj,
                bq=layer.self_attn.q_proj.bias,
                bk=layer.self_attn.k_proj.bias,
                bv=layer.self_attn.v_proj.bias,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=w_input_layernorm,
                w_post_attention_layernorm=w_post_attention_layernorm,
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta,
            ))

        self.norm = RMSNorm(
            mlx_model.args.hidden_size,
            weight=mlx_model.model.norm.weight.astype(precision),
            eps=mlx_model.args.rms_norm_eps,
        )
        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head).astype(precision)
        else:
            self.w_lm_head = None
        

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        embed = self.embedding(inputs)
        for layer in self.transformer_blocks:
            embed = layer(embed, mask="causal")
        embed = self.norm(embed)
        if self.w_lm_head is not None:
            return linear(embed, self.w_lm_head)
        else:
            return self.embedding.as_linear(embed)
