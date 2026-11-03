# Territory: other-kernels

- **Owner:** explorer-other-kernels
- **Status:** exploring
- **Version:** 1
- **Updated:** 2026-11-03T21:20Z
- **Next:** read sort/reduce/binary/unary/copy/indexing kernel headers; map routing+combine launches

## What it does

The routing and combine slots are built from MLX's generic Metal kernels, not the
quantized ones: a small matmul plus sigmoid for router scores, add/negate,
`argpartition` top-4, `take_along_axis` gather, row reductions, broadcast
multiply/divide, astype, and add. This territory maps those kernels in
`.venv/.../mlx/backend/metal/kernels/` to the slot ops that launch them, with
shapes, bytes, and launch structure at 1, 8 and 32 rows.

## Checklist

(to fill per piece)

## Relations

- Fed by: `model` / `switch-layers` (MoE.__call__ launches these ops); `quantized-kernels` feeds combine's `y` input.
- Feeds: routing's `inds`/`weights` feed `quantized-kernels` (experts) and combine; combine's `out` feeds the next layer.

## Inefficiencies

- none yet

## Files read

- none yet

## From other agents

(none yet)
