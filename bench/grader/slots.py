"""The parts of MobileMoE-S's MoE block that a Muse Spark kernel may replace.

Each slot has a reference (the port's own code, split as in breakdown.py),
named inputs and outputs (claude/muse-agent-brief.md), and a way to install
an implementation into MoE.__call__ for every layer. build() turns a launch
spec (data, never code) into mx.fast.metal_kernel calls.

Record the grader's inputs from a real decode step (plain MLX, no API):

  .venv/bin/python bench/grader/slots.py capture
"""

import re
from pathlib import Path

import mlx.core as mx
import numpy as np

D, E, K, H = 768, 60, 4, 384
SCALE = 2.5  # routed_scaling_factor; norm_topk_prob is true
NAMES = {"D": D, "E": E, "K": K, "H": H}
DTYPES = {
    "float32": mx.float32,
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "uint32": mx.uint32,
    "int32": mx.int32,
}
SLOTS = {
    "routing": (["x", "w_router", "bias"], ["inds", "weights"]),
    "combine": (["y", "weights", "shared"], ["out"]),
}
EXPERT_WEIGHTS = [
    f"{p}_{t}" for p in ("gate", "up", "down") for t in ("w", "scales", "biases")
]
SLOTS["experts"] = (["x", "inds"] + EXPERT_WEIGHTS, ["y"])
SLOTS["routing+combine"] = (
    SLOTS["routing"][0] + ["y", "shared"],
    ["inds", "weights", "out"],
)
INPUTS = Path(__file__).resolve().parents[2] / "models" / "muse-inputs"


def ref_routing(x, w_router, bias):
    scores = mx.sigmoid(x.astype(mx.float32) @ w_router.T)
    inds = mx.argpartition(-(scores + bias), kth=K - 1, axis=-1)[:, :K]
    w = mx.take_along_axis(scores, inds, axis=-1)
    return inds, w / (w.sum(axis=-1, keepdims=True) + 1e-20) * SCALE


def ref_combine(y, weights, shared):
    return (y * weights[..., None]).sum(axis=-2).astype(mx.bfloat16) + shared


def ref_experts(x, inds, *weights):
    """The port's SwitchGLU on flat rows: [rows, D] and [rows, K] -> [rows, K, D]."""
    from mlx_lm.models.activations import swiglu
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    gate, up, down = weights[0:3], weights[3:6], weights[6:9]
    xx = mx.expand_dims(x, (-2, -3))
    do_sort = inds.size >= 64
    idx, inv = inds, None
    if do_sort:
        xx, idx, inv = _gather_sort(xx, inds)

    def qmm(a, w):
        return mx.gather_qmm(
            a,
            *w,
            rhs_indices=idx,
            transpose=False,
            group_size=32,
            bits=4,
            mode="affine",
            sorted_indices=do_sort,
        )

    y = qmm(swiglu(qmm(xx, gate), qmm(xx, up)), down)
    if do_sort:
        y = _scatter_unsort(y, inv, inds.shape)
    return y.squeeze(-2)


REFERENCE = {"routing": ref_routing, "combine": ref_combine, "experts": ref_experts}


def expert_weights(experts):
    """A layer's SwitchGLU weights, named as in the experts slot."""
    return {
        f"{p}_{t}": getattr(experts, f"{p}_proj")[k]
        for p in ("gate", "up", "down")
        for t, k in (("w", "weight"), ("scales", "scales"), ("biases", "biases"))
    }


def make_moe_call(routing=ref_routing, combine=ref_combine, experts=None):
    """MoE.__call__ with its slots swapped for *routing*, *combine*, *experts*.

    With the references it gives the port's outputs bit for bit (checked by
    check_identity), so any difference comes from the swapped slot. With
    experts=None the port's own SwitchGLU runs.
    """

    def moe_call(self, x):
        lead = x.shape[:-1]
        flat = x.reshape(-1, D)
        bias = self.expert_bias.astype(mx.float32)
        inds, weights = routing(flat, self.router.weight, bias)
        if experts is None:
            y = self.experts(x, inds.reshape(*lead, K))
        else:
            w = expert_weights(self.experts)
            y = experts(flat, inds, *[w[n] for n in EXPERT_WEIGHTS])
        shared = self.shared_expert(x)
        out = combine(y.reshape(-1, K, D), weights, shared.reshape(-1, D))
        return out.reshape(*lead, D)

    return moe_call


def check_identity(model):
    """The split MoE with reference slots equals the port's, at 1, 8, 32 rows."""
    from mlx_lm.models import mobilemoe

    original = mobilemoe.MoE.__call__
    for split in (make_moe_call(), make_moe_call(experts=ref_experts)):
        for rows in (1, 8, 32):
            moe = model.model.layers[3].feed_forward
            x = mx.random.normal((1, rows, D)).astype(mx.bfloat16)
            if not mx.array_equal(original(moe, x), split(moe, x)).item():
                raise AssertionError(
                    f"split MoE differs from the port's at {rows} rows"
                )


def capture(model, rows, step=8, seed=None):
    """Every layer's slot inputs and reference outputs in one real decode step."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "micro"))
    import breakdown as bd
    from mlx_lm.models import mobilemoe
    from mlx_lm.models.cache import make_prompt_cache

    original = mobilemoe.MoE.__call__
    seen = []

    def recording_routing(x, w_router, bias):
        inds, weights = ref_routing(x, w_router, bias)
        seen.append({"x": x, "w_router": w_router, "bias": bias, "inds": inds})
        return inds, weights

    def recording_combine(y, weights, shared):
        seen[-1].update(y=y, weights_in=weights, shared=shared)
        return ref_combine(y, weights, shared)

    mx.random.seed(rows if seed is None else seed)
    ids = mx.random.randint(0, 100000, (rows, 128))
    cache = make_prompt_cache(model)
    y = bd.prefill(model, ids, cache)
    for _ in range(step - 1):
        y = bd.greedy(model(y, cache=cache))
        mx.eval(y)
    mobilemoe.MoE.__call__ = make_moe_call(recording_routing, recording_combine)
    try:
        logits = model(y, cache=cache)
        mx.eval(logits, [list(s.values()) for s in seen])
    finally:
        mobilemoe.MoE.__call__ = original
    return seen


def save_inputs(model):
    INPUTS.mkdir(parents=True, exist_ok=True)
    for rows in (1, 8, 32):
        layers = capture(model, rows)
        arrays = {
            f"L{i:02d}_{k}": np.array(
                v.astype(mx.float32) if v.dtype == mx.bfloat16 else v
            )
            for i, layer in enumerate(layers)
            for k, v in layer.items()
        }
        np.savez(INPUTS / f"rows{rows}.npz", **arrays)
        print(f"captured {len(layers)} layers at {rows} rows", flush=True)


def load_model():
    """The served checkpoint, with its weights in memory."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "micro"))
    import breakdown as bd
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(__file__).resolve().parents[2] / bd.MODEL)
    mx.eval(model.parameters())
    return model


def model_expert_weights(model):
    return [expert_weights(layer.feed_forward.experts) for layer in model.model.layers]


def load_expert_weights():
    """Every layer's SwitchGLU weights, from the served checkpoint."""
    return model_expert_weights(load_model())


PROMPT_REQUESTS = (1, 8, 32)


def capture_prefill(model, requests, with_weights=None, tokens=None):
    """Every layer's slot inputs in the first chunk of a prefill, as load_inputs.

    The same MoE.__call__ runs the prefill. With 128-token prompts and
    breakdown.py's 2,048-token chunks, 1, 8 and 32 requests give the slots
    127, 1,016 and 2,048 rows; from 60 rows MLX's own experts kernel changes
    (gather_qmm_rhs). The prompts are time_candidates.py's. *tokens* sets the
    chunk's length per request instead, for requests x tokens rows.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "micro"))
    import breakdown as bd
    from mlx_lm.models import mobilemoe
    from mlx_lm.models.cache import make_prompt_cache

    seen = []

    def routing(x, w_router, bias):
        inds, weights = ref_routing(x, w_router, bias)
        seen.append({"x": x, "w_router": w_router, "bias": bias, "inds": inds})
        return inds, weights

    def combine(y, weights, shared):
        seen[-1].update(y=y, weights=weights, shared=shared)
        return ref_combine(y, weights, shared)

    mx.random.seed(requests)
    ids = mx.random.randint(0, 100000, (requests, max(128, (tokens or 0) + 1)))
    chunk = tokens or min(bd.PREFILL_TOKENS // requests, 127)
    original = mobilemoe.MoE.__call__
    mobilemoe.MoE.__call__ = make_moe_call(routing, combine)
    try:
        model(ids[:, :chunk], cache=make_prompt_cache(model))
    finally:
        mobilemoe.MoE.__call__ = original
    for i, layer in enumerate(seen):
        layer["inds"] = layer["inds"].astype(mx.uint32)
        if with_weights is not None:
            layer.update(with_weights[i])
    mx.eval(seen)
    return seen


def load_inputs(rows, with_weights=None):
    """[{name: mx.array}] per layer, with bfloat16 restored where the model had it.

    *with_weights*, from load_expert_weights or model_expert_weights, adds each layer's expert weights.
    """
    data = np.load(INPUTS / f"rows{rows}.npz")
    bf16 = {"x", "y", "shared"}
    layers = {}
    for key in data.files:
        i, name = int(key[1:3]), key[4:]
        a = mx.array(data[key])
        layers.setdefault(i, {})[name] = a.astype(mx.bfloat16) if name in bf16 else a
    for i, layer in layers.items():
        layer["weights"] = layer.pop("weights_in")
        if "inds" in layer:
            layer["inds"] = layer["inds"].astype(mx.uint32)
        if with_weights is not None:
            layer.update(with_weights[i])
    return [layers[i] for i in sorted(layers)]


# Muse Spark's launch spec: integers, or products of integers and NAMES/rows.
TERM = re.compile(r"^(\d+|rows|D|E|K|H)$")


def dim(expr, rows):
    if isinstance(expr, bool) or not isinstance(expr, (int, str)):
        raise ValueError(f"bad size {expr!r}")
    if isinstance(expr, int):
        return expr
    value = 1
    for term in expr.replace(" ", "").split("*"):
        if not TERM.match(term):
            raise ValueError(f"bad size term {term!r} in {expr!r}")
        value *= rows if term == "rows" else NAMES.get(term) or int(term)
    return value


def template_value(v):
    if isinstance(v, bool) or isinstance(v, int):
        return v
    if isinstance(v, str) and v in DTYPES:
        return DTYPES[v]
    raise ValueError(f"bad template value {v!r}")


def build(spec, sources, slot):
    """A function slot_inputs -> slot_outputs from Muse Spark's spec and Metal blocks.

    With "max_rows" in the spec, inputs with more rows than that run the slot's
    reference under mx.compile instead of the kernels, so an attempt can keep
    prompt-sized inputs on MLX's own code (claude/adr/0011-decode-and-prefill.md).
    """
    slot_in, slot_out = SLOTS[slot]
    kernels = []
    for k in spec["kernels"]:
        header = "" if k.get("header_block") is None else sources[k["header_block"]]
        fn = mx.fast.metal_kernel(
            name=re.sub(r"\W", "_", str(k["name"])),
            input_names=list(k["inputs"]),
            output_names=[o["name"] for o in k["outputs"]],
            source=sources[k["source_block"]],
            header=header,
            atomic_outputs=bool(k.get("atomic_outputs", False)),
        )
        kernels.append((k, fn))
    missing = set(slot_out) - set(spec["slot_outputs"])
    if missing:
        raise ValueError(f"slot_outputs is missing {sorted(missing)}")
    max_rows = spec.get("max_rows")
    if max_rows is not None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError(f"bad max_rows {max_rows!r}")
        compiled = mx.compile(REFERENCE[slot])

    def run(**inputs):
        rows = inputs[slot_in[0]].shape[0]
        if max_rows is not None and rows > max_rows:  # the compiled reference takes over
            out = compiled(*[inputs[n] for n in slot_in])
            return dict(zip(slot_out, out if isinstance(out, (tuple, list)) else [out]))
        env = dict(inputs)
        for k, fn in kernels:
            tg = [int(t) for t in k["threadgroup"]]
            if any(t < 1 for t in tg) or tg[0] * tg[1] * tg[2] > 1024:
                raise ValueError(f"bad threadgroup {tg}")
            outs = fn(
                inputs=[env[n] for n in k["inputs"]],
                template=[
                    (n, template_value(v)) for n, v in (k.get("template") or {}).items()
                ],
                grid=tuple(dim(g, rows) for g in k["grid"]),
                threadgroup=tuple(tg),
                output_shapes=[
                    tuple(dim(s, rows) for s in o["shape"]) for o in k["outputs"]
                ],
                output_dtypes=[DTYPES[o["dtype"]] for o in k["outputs"]],
                init_value=k.get("init_value"),
            )
            env.update(zip([o["name"] for o in k["outputs"]], outs))
        return {name: env[src] for name, src in spec["slot_outputs"].items()}

    return run


def main():
    import sys

    if sys.argv[1:] != ["capture"]:
        raise SystemExit("usage: slots.py capture")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "micro"))
    import breakdown as bd
    import gpu_lock

    gpu_lock.hold("slots.py capture")
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(__file__).resolve().parents[2] / bd.MODEL)
    mx.eval(model.parameters())
    check_identity(model)
    print("split MoE with reference slots: bit-identical to the port's")
    save_inputs(model)


if __name__ == "__main__":
    main()
