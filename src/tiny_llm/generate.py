import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        y = mx.expand_dims(y, 0)
        # print("Y Shape", y.shape)
        logits = model(y)[:, -1, :]

        # log sum exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - mx.logsumexp(logits, keepdims=True)

        if sampler is None:
            return mx.argmax(logits, axis=-1)
        else:
            return sampler(logits)

    curr_tokens = tokenizer.encode(prompt)

    print(prompt, end="", flush=True)

    while curr_tokens[-1] != tokenizer.eos_token_id:
        # print("Current Tokens", curr_tokens)
        next_token = _step(model, mx.array(curr_tokens))
        curr_tokens.append(next_token.item())

        tokenizer.detokenizer.add_token(next_token.item())
        # print(tokenizer.detokenizer.decode(curr_tokens), end="", flush=True)
        print(tokenizer.detokenizer.last_segment, end="", flush=True)
    return tokenizer.detokenizer.text

def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
