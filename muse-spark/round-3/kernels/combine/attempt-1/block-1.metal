// Rounding-barrier helpers for the fused combine kernel (block-0.metal).
//
// Each helper performs exactly one floating-point op in its own
// non-inlined call. Because the mul/add lives in a separate function
// that the compiler must not inline, its result rounds once to float32
// at the return, and no mul can be FP-contracted into a neighbouring
// add: there is no expression tree spanning the call boundary. Values
// cross the boundary in registers, so unlike the starting point's
// seven `volatile float` barriers this issues no thread-private memory
// traffic. The fold order, the bfloat16 roundings, and every other
// line of the kernel are unchanged (see block-0.metal).
__attribute__((noinline)) float combine_mul(float a, float b) {
    return a * b;
}
__attribute__((noinline)) float combine_add(float a, float b) {
    return a + b;
}
