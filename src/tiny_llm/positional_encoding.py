from re import M
import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        
        self.half_dims = dims // 2
        self.cos_freqs, self.sin_freqs = self._compute_freqs()


    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # x shape -> (N, L, H, D)
        # don't need to consider offset as list[slice]

        if self.traditional:
            return self._compute_rope_traditional(x, offset)
        else:
            return self._compute_rope_non_traditional(x, offset)

    def _compute_rope_traditional(self, x: mx.array, offset: list[slice] | slice | None = None) -> mx.array:

        curr_offset = offset.start if offset is not None else 0
        seq_len = x.shape[1]

        # x shape -> (N, L, H, D)
        # output shape -> (N, L, H, D)
        output = mx.zeros_like(x)
        for j in range(seq_len):
            cos_freqs = self.cos_freqs[curr_offset, :] 
            sin_freqs = self.sin_freqs[curr_offset, :]
            
            curr_x = x[:, j, :, :]

            for i in range(self.half_dims):

                output[..., j, :, 2*i] = curr_x[..., 2*i] * cos_freqs[i] - curr_x[..., 2*i+1] * sin_freqs[i]
                output[..., j, :, 2*i+1] = curr_x[..., 2*i] * sin_freqs[i] + curr_x[..., 2*i + 1] * cos_freqs[i]

            curr_offset += 1

        return output

    def _compute_rope_non_traditional(self, x: mx.array, offset: list[slice] | slice | None = None) -> mx.array:

        curr_offset = offset.start if offset is not None else 0
        seq_len = x.shape[1]

        # x shape -> (N, L, H, D)
        # output shape -> (N, L, H, D)
        output = mx.zeros_like(x)
        for j in range(seq_len):
            cos_freqs = self.cos_freqs[curr_offset, :] 
            sin_freqs = self.sin_freqs[curr_offset, :]
            
            curr_x = x[:, j, :, :]

            for i in range(self.half_dims):

                output[..., j, :, i] = curr_x[..., i] * cos_freqs[i] - curr_x[..., self.half_dims+i] * sin_freqs[i]
                output[..., j, :, self.half_dims + i] = curr_x[..., i] * sin_freqs[i] + curr_x[..., self.half_dims + i] * cos_freqs[i]

            curr_offset += 1

        return output


    # returns cos_freqs and sin_freqs
    def _compute_freqs(self) -> (mx.array, mx.array):
        # shape -> (MAX_SEQ_LEN, D // 2) which is essentially offest/position 0 to MAX_SEQ_LEN - 1
        inner = (2 * mx.arange(0, self.half_dims, dtype=mx.float32)) / self.dims
        # freqs shape -> (D // 2)
        freqs = mx.power(self.base, -inner)

        # m shape -> (MAX_SEQ_LEN)
        m = mx.arange(self.seq_len)
    
        # freqs shape -> (MAX_SEQ_LEN, D // 2)
        thetas = mx.outer(m, freqs)
        cos_freqs = mx.cos(thetas)
        sin_freqs = mx.sin(thetas)
        return cos_freqs, sin_freqs