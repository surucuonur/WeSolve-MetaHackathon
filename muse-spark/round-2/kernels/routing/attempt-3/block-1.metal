// Header for hyp-2-routing-exact hardening, round-2 attempt-3.
// Only simd shuffles remain (GEMV tree); the simdgroup matrix include of
// attempt-1 is gone with the MMA staging it served.
#include <metal_simdgroup>
