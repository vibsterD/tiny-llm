import mlx.core as mx
import copy

from numpy import cumsum


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if top_k is not None and top_k > 0:
            elements_to_mask = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[:, top_k:]
            logprobs[:, elements_to_mask] = -mx.inf
        
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            cumsum_prob = mx.cumsum(mx.exp(logprobs[:, sorted_idx]), axis=-1)
            mask_elements = cumsum_prob < top_p
            mask_elements[..., 0] = True
            logprobs[:, sorted_idx] = mx.where(mask_elements, logprobs[:, sorted_idx], -mx.inf)
        
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        
        logprobs = logprobs / temp
        return mx.random.categorical(logprobs, axis=-1)

    return sample
