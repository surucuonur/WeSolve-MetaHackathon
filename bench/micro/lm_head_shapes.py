"""Investigation: does the output layer cost more on vLLM's row layout?

vllm-metal passes the model one packed sequence, so at 32 decode rows the
output layer gets hidden states shaped (1, 32, 768); plain MLX passes
(32, 1, 768). This times the same product on both layouts, interleaved, and
checks that the logits match.

  .venv/bin/python bench/micro/lm_head_shapes.py
"""

import random
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_model

model, _ = load_model(Path("models/MobileMoE-S-QAT-mlx-bfloat16"))
mx.eval(model.parameters())
head = model.model.embed_tokens  # the output layer is tied to the embedding
plain = mx.random.normal((32, 1, 768)).astype(mx.bfloat16)
packed = plain.reshape(1, 32, 768)
mx.eval(plain, packed)
same = mx.array_equal(
    head.as_linear(plain).reshape(1, 32, -1), head.as_linear(packed)
).item()
print("same logits:", same)


def per_call(x, n=100):
    ys = []
    t0 = time.perf_counter()
    for _ in range(n):
        ys.append(head.as_linear(x))
        mx.async_eval(ys[-1])
    mx.eval(ys)
    return (time.perf_counter() - t0) / n * 1e3


times = {"(32, 1, 768)": [], "(1, 32, 768)": []}
for rep in range(6):
    order = [("(32, 1, 768)", plain), ("(1, 32, 768)", packed)]
    random.Random(rep).shuffle(order)
    for name, x in order:
        per_call(x, 10)  # warm-up
        times[name].append(per_call(x))
for name, v in times.items():
    print(
        f"{name}: median {statistics.median(v):.3f} ms per call "
        f"[{min(v):.3f}-{max(v):.3f}]"
    )
