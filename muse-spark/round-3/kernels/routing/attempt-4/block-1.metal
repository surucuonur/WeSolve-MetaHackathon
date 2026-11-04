// Header for hyp-3-routing-prefill direction (b), round-3 attempt-2.
// Kernel 0 keeps attempt-4's GEMV shuffle tree; kernel 1 reuses this
// header so metal::precise stays available exactly as in attempt-4.
#include <metal_simdgroup>
