"""Check the hand-off to the compiled reference (min_rows, max_rows) and how the grader judges it.

slots.build with a toy combine kernel that copies `shared`: inputs inside
[min_rows, max_rows] must run it, the others the compiled reference, bit for
bit; bad bounds must be refused. judge() must leave "handed off" sizes out of
a phase (claude/adr/0015-sets-and-hand-off-below.md). Uses the GPU for a few
small kernels, not the model.

  .venv/bin/python bench/grader/test_hand_off.py
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slots  # noqa: E402
from check_candidate import judge  # noqa: E402

COPY = "uint i = thread_position_in_grid.x;\nout[i] = shared[i];\n"


def toy(**bounds):
    spec = {
        "kernels": [{
            "name": "copy_shared", "inputs": ["shared"], "source_block": 0,
            "grid": ["rows*D", 1, 1], "threadgroup": [256, 1, 1],
            "outputs": [{"name": "out", "shape": ["rows", "D"], "dtype": "bfloat16"}],
        }],
        "slot_outputs": {"out": "out"},
        **bounds,
    }
    return slots.build(spec, [COPY], "combine")


def inputs(rows):
    mx.random.seed(rows)
    return {
        "y": mx.random.normal((rows, slots.K, slots.D)).astype(mx.bfloat16),
        "weights": mx.random.uniform(shape=(rows, slots.K)),
        "shared": mx.random.normal((rows, slots.D)).astype(mx.bfloat16),
    }


def runs(fn, rows):
    """"kernel" or "compiled": which code produced the output, checked bit for bit."""
    x = inputs(rows)
    out = fn(**x)["out"]
    if mx.array_equal(out, x["shared"]).item():
        return "kernel"
    ref = mx.compile(slots.ref_combine)(x["y"], x["weights"], x["shared"])
    assert mx.array_equal(out, ref).item(), f"{rows} rows: neither the kernel's nor the reference's output"
    return "compiled"


def main():
    failed = []

    def expect(name, got, want):
        print(f"{'ok  ' if got == want else 'FAIL'} {name}: {got}")
        if got != want:
            failed.append(name)

    for bounds, want in [
        ({}, {1: "kernel", 8: "kernel", 127: "kernel"}),
        ({"max_rows": 32}, {1: "kernel", 32: "kernel", 33: "compiled", 127: "compiled"}),
        ({"min_rows": 2}, {1: "compiled", 2: "kernel", 127: "kernel"}),
        ({"min_rows": 8, "max_rows": 59}, {1: "compiled", 7: "compiled", 8: "kernel", 59: "kernel", 60: "compiled"}),
    ]:
        fn = toy(**bounds)
        got = {rows: runs(fn, rows) for rows in want}
        expect(f"build {bounds or 'no bounds'}", got, want)
        expect(f"hands_off {bounds or 'no bounds'}", {r: fn.hands_off(r) for r in want}, {r: w == "compiled" for r, w in want.items()})

    for bounds in [{"min_rows": 0}, {"min_rows": True}, {"min_rows": 2.0}, {"min_rows": 33, "max_rows": 32}]:
        try:
            toy(**bounds)
            got = "accepted"
        except ValueError as e:
            got = "refused" if str(e) else "refused, no reason"
        expect(f"build {bounds}", got, "refused")

    d, p = "rows ", "prefill rows "
    for name, verdicts, want in [
        ("all timed", {d + "1": "faster", d + "8": "faster", d + "32": "faster", p + "127": "same", p + "1016": "same", p + "2048": "slower"},
         ({"decode": "faster", "prefill": "slower"}, False)),
        ("prefill handed off", {d + "1": "faster", d + "8": "faster", d + "32": "faster", p + "127": "handed off", p + "1016": "handed off", p + "2048": "handed off"},
         ({"decode": "faster", "prefill": "handed off"}, True)),
        ("1 row handed off too", {d + "1": "handed off", d + "8": "faster", d + "32": "faster", p + "127": "handed off", p + "1016": "handed off", p + "2048": "handed off"},
         ({"decode": "faster", "prefill": "handed off"}, True)),
        ("a timed size not faster", {d + "1": "handed off", d + "8": "faster", d + "32": "same", p + "127": "handed off", p + "1016": "handed off", p + "2048": "handed off"},
         ({"decode": "not slower", "prefill": "handed off"}, False)),
        ("everything handed off", {d + "1": "handed off", d + "8": "handed off", d + "32": "handed off", p + "127": "handed off", p + "1016": "handed off", p + "2048": "handed off"},
         ({"decode": "handed off", "prefill": "handed off"}, False)),
    ]:
        expect(f"judge {name}", judge(verdicts), want)

    print(f"\n{'all passed' if not failed else f'{len(failed)} failed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
