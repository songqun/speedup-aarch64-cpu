# speedup-aarch64-cpu
A computing kernel implementation in ML inference framework aiming at theoretical limit on ARMv8 CPU in single thread.

## General Idea

* SIMD
* Packing
* Blocking
* Instruction Reorder

TODO: More description in detail

## Trick

On Cortex-A55, q-form <b>fmla</b> and <b>ldr</b> cannot be dual issue, so instruction reorder cannot fully hide <b>ldr</b> with <b>fmla</b>.  
However, I find a trick to split q-form <b>ldr</b> into 3 instructions and all of them can be hidden by <b>fmla</b>.  
So we need a computing kernel with computing / load ratio > 3.

The kernel in this project is 

sgemm_4x16: 4\*K matrix multiples K\*16 matrix.  
Computing / load ratio is: (4(4\*1 weight)\*4(1 in)) / (1(4\*1 in) + 4(4\*1 weight)) = 16/5 > 3.

Or 

sgemm_8x8: 8\*K matrix multiples K\*8 matrix.  
Computing / load ratio is: (2(4\*1 weight)\*8(1 in)) / (2(4\*1 in) + 2(4\*1 weight)) = 4 > 3