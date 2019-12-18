#include "sgemm.hpp"


void sgemm_8x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k)
{
#if defined(_USE_NEON_A55)
  __asm__ __volatile__(
    "ldr q24, [%[b_0]]\n"         // b_O0o[0:3]
    "ldr q25, [%[b_1]]\n"         // b_O1o[0:3]
    "mov x0, %[k]\n"              // K (ic_blk)
    "mov v4.16b, v24.16b\n"       // out_O0HW0o[0:3]
    "ldr d0, [%[in]]\n"           // in_C0hw[0:1]
    "mov v5.16b, v24.16b\n"       // out_O0HW1o[0:3]
    "ldr x1, [%[in], #8]\n"       // in_C0hw[2:3]
    "mov v6.16b, v24.16b\n"       // out_O0HW2o[0:3]
    "ins v0.d[1], x1\n"           // in_C0hw[0:3]
    "mov v7.16b, v24.16b\n"       // out_O0HW3o[0:3]
    "ldr d1, [%[in], #16]\n"      // in_C0hw[4:5]
    "mov v8.16b, v24.16b\n"       // out_O0HW4o[0:3]
    "ldr x2, [%[in], #24]\n"      // in_C0hw[6:7]
    "mov v9.16b, v24.16b\n"       // out_O0HW5o[0:3]
    "ins v1.d[1], x2\n"           // in_C0hw[4:7]
    "mov v10.16b, v24.16b\n"      // out_O0HW6o[0:3]
    "ldr d20, [%[wgt]]\n"         // wgt_C0O0o[0:1]
    "mov v11.16b, v24.16b\n"      // out_O0HW7o[0:3]
    "ldr x3, [%[wgt], #8]\n"      // wgt_C0O0o[2:3]
    "mov v12.16b, v25.16b\n"      // out_O1HW0o[0:3]
    "ins v20.d[1], x3\n"          // wgt_C0O0o[0:3]
    "mov v13.16b, v25.16b\n"      // out_O1HW1o[0:3]
    "ldr d21, [%[wgt], #16]\n"    // wgt_C0O1o[0:1]
    "mov v14.16b, v25.16b\n"      // out_O1HW2o[0:3]
    "ldr x4, [%[wgt], #24]\n"     // wgt_C0O1o[2:3]
    "mov v15.16b, v25.16b\n"      // out_O1HW3o[0:3]
    "ins v21.d[1], x4\n"          // wgt_C0O1o[0:3]
    "mov v16.16b, v25.16b\n"      // out_O1HW4o[0:3]
    "mov v17.16b, v25.16b\n"      // out_O1HW5o[0:3]
    "mov v18.16b, v25.16b\n"      // out_O1HW6o[0:3]
    "mov v19.16b, v25.16b\n"      // out_O1HW7o[0:3]
    "0:\n"
    "ldr d2, [%[in], #32]\n"      // in_C1hw[0:1]
    "fmla v4.4s, v20.4s, v0.s[0]\n"
    "ldr x1, [%[in], #40]\n"      // in_C1hw[2:3]
    "fmla v5.4s, v20.4s, v0.s[1]\n"
    "ins v2.d[1], x1\n"           // in_C1hw[0:3]
    "fmla v6.4s, v20.4s, v0.s[2]\n"
    "ldr d3, [%[in], #48]\n"      // in_C1hw[4:5]
    "fmla v7.4s, v20.4s, v0.s[3]\n"
    "ldr x2, [%[in], #56]\n"      // in_C1hw[6:7]
    "fmla v8.4s, v20.4s, v1.s[0]\n"
    "ins v3.d[1], x2\n"           // in_C1hw[4:7]
    "fmla v9.4s, v20.4s, v1.s[1]\n"
    "ldr d22, [%[wgt], #32]\n"    // wgt_C1O0o[0:1]
    "fmla v10.4s, v20.4s, v1.s[2]\n"
    "ldr x3, [%[wgt], #40]\n"     // wgt_C1O0o[2:3]
    "fmla v11.4s, v20.4s, v1.s[3]\n"
    "ins v22.d[1], x3\n"          // wgt_C1O0o[0:3]
    "fmla v12.4s, v21.4s, v0.s[0]\n"
    "ldr d23, [%[wgt], #48]\n"    // wgt_C1O1o[0:1]
    "fmla v13.4s, v21.4s, v0.s[1]\n"
    "ldr x4, [%[wgt], #56]\n"     // wgt_C1O1o[2:3]
    "fmla v14.4s, v21.4s, v0.s[2]\n"
    "ins v23.d[1], x4\n"          // wgt_C1O1o[0:3]
    "fmla v15.4s, v21.4s, v0.s[3]\n"
    "fmla v16.4s, v21.4s, v1.s[0]\n"
    "fmla v17.4s, v21.4s, v1.s[1]\n"
    "fmla v18.4s, v21.4s, v1.s[2]\n"
    "fmla v19.4s, v21.4s, v1.s[3]\n"

    "ldr d0, [%[in], #64]\n"      // in_C0hw[0:1]
    "fmla v4.4s, v22.4s, v2.s[0]\n"
    "ldr x1, [%[in], #72]\n"      // in_C0hw[2:3]
    "fmla v5.4s, v22.4s, v2.s[1]\n"
    "ins v0.d[1], x1\n"           // in_C0hw[0:3]
    "fmla v6.4s, v22.4s, v2.s[2]\n"
    "ldr d1, [%[in], #80]\n"      // in_C0hw[4:5]
    "fmla v7.4s, v22.4s, v2.s[3]\n"
    "ldr x2, [%[in], #88]\n"      // in_C0hw[6:7]
    "fmla v8.4s, v22.4s, v3.s[0]\n"
    "ins v1.d[1], x2\n"           // in_C0hw[4:7]
    "fmla v9.4s, v22.4s, v3.s[1]\n"
    "ldr d20, [%[wgt], #64]\n"    // wgt_C0O0o[0:1]
    "fmla v10.4s, v22.4s, v3.s[2]\n"
    "ldr x3, [%[wgt], #72]\n"     // wgt_C0O0o[2:3]
    "fmla v11.4s, v22.4s, v3.s[3]\n"
    "ins v20.d[1], x3\n"          // wgt_C0O0o[0:3]
    "fmla v12.4s, v23.4s, v2.s[0]\n"
    "ldr d21, [%[wgt], #80]\n"    // wgt_C0O1o[0:1]
    "fmla v13.4s, v23.4s, v2.s[1]\n"
    "ldr x4, [%[wgt], #88]\n"     // wgt_C0O1o[2:3]
    "fmla v14.4s, v23.4s, v2.s[2]\n"
    "ins v21.d[1], x4\n"          // wgt_C0O1o[0:3]
    "fmla v15.4s, v23.4s, v2.s[3]\n"
    "add %[in], %[in], #64\n"
    "fmla v16.4s, v23.4s, v3.s[0]\n"
    "add %[wgt], %[wgt], #64\n"
    "fmla v17.4s, v23.4s, v3.s[1]\n"
    "subs x0, x0, #2\n"
    "fmla v18.4s, v23.4s, v3.s[2]\n"
    "fmla v19.4s, v23.4s, v3.s[3]\n"
    "bne 0b\n"
    "str q4, [%[out_0]]\n"        // out_O0HW0o[0:3]
    "str q5, [%[out_0], #16]\n"   // out_O0HW1o[0:3]
    "str q6, [%[out_0], #32]\n"   // out_O0HW2o[0:3]
    "str q7, [%[out_0], #48]\n"   // out_O0HW3o[0:3]
    "str q8, [%[out_0], #64]\n"   // out_O0HW4o[0:3]
    "str q9, [%[out_0], #80]\n"   // out_O0HW5o[0:3]
    "str q10, [%[out_0], #96]\n"  // out_O0HW6o[0:3]
    "str q11, [%[out_0], #112]\n" // out_O0HW7o[0:3]
    "str q12, [%[out_1]]\n"       // out_O1HW0o[0:3]
    "str q13, [%[out_1], #16]\n"  // out_O1HW1o[0:3]
    "str q14, [%[out_1], #32]\n"  // out_O1HW2o[0:3]
    "str q15, [%[out_1], #48]\n"  // out_O1HW3o[0:3]
    "str q16, [%[out_1], #64]\n"  // out_O1HW4o[0:3]
    "str q17, [%[out_1], #80]\n"  // out_O1HW5o[0:3]
    "str q18, [%[out_1], #96]\n"  // out_O1HW6o[0:3]
    "str q19, [%[out_1], #112]\n" // out_O1HW7o[0:3]
    :[out_0]"+r"(out_0),
     [out_1]"+r"(out_1),
     [in]"+r"(in),
     [wgt]"+r"(wgt)
    :[k]"r"(k),
     [b_0]"r"(b_0),
     [b_1]"r"(b_1)
    :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23", "q24", "q25", "x0", "x1", "x2", "x3", "x4"
  );
#elif defined(_USC_NEON_A76)
  __asm__ __volatile__(
    "ldr q24, [%[b_0]]\n"         // b_O0o[0:3]
    "ldr q25, [%[b_1]]\n"         // b_O1o[0:3]
    "mov x0, %[k]\n"              // K (ic_blk)
    "mov v4.16b, v24.16b\n"       // out_O0HW0o[0:3]
    "mov v5.16b, v24.16b\n"       // out_O0HW1o[0:3]
    "mov v6.16b, v24.16b\n"       // out_O0HW2o[0:3]
    "mov v7.16b, v24.16b\n"       // out_O0HW3o[0:3]
    "mov v8.16b, v24.16b\n"       // out_O0HW4o[0:3]
    "mov v9.16b, v24.16b\n"       // out_O0HW5o[0:3]
    "mov v10.16b, v24.16b\n"      // out_O0HW6o[0:3]
    "mov v11.16b, v24.16b\n"      // out_O0HW7o[0:3]
    "mov v12.16b, v25.16b\n"      // out_O1HW0o[0:3]
    "mov v13.16b, v25.16b\n"      // out_O1HW1o[0:3]
    "mov v14.16b, v25.16b\n"      // out_O1HW2o[0:3]
    "mov v15.16b, v25.16b\n"      // out_O1HW3o[0:3]
    "mov v16.16b, v25.16b\n"      // out_O1HW4o[0:3]
    "mov v17.16b, v25.16b\n"      // out_O1HW5o[0:3]
    "mov v18.16b, v25.16b\n"      // out_O1HW6o[0:3]
    "mov v19.16b, v25.16b\n"      // out_O1HW7o[0:3]
    "ldr q0, [%[in]]\n"           // in_C0hw[0:3]
    "ldr q1, [%[in], #16]\n"      // in_C0hw[4:7]
    "ldr q20, [%[wgt]]\n"         // wgt_C0O0o[0:3]
    "ldr q21, [%[wgt], #16]\n"    // wgt_C0O1o[0:3]
    "0:\n"
    "ldr q2, [%[in], #32]\n"      // in_C1hw[0:3]
    "ldr q3, [%[in], #48]\n"      // in_C1hw[4:7]
    "ldr q22, [%[wgt], #32]\n"    // wgt_C1O0o[0:3]
    "ldr q23, [%[wgt], #48]\n"    // wgt_C1O1o[0:3]
    "fmla v4.4s, v20.4s, v0.s[0]\n"
    "fmla v5.4s, v20.4s, v0.s[1]\n"
    "fmla v6.4s, v20.4s, v0.s[2]\n"
    "fmla v7.4s, v20.4s, v0.s[3]\n"
    "fmla v8.4s, v20.4s, v1.s[0]\n"
    "fmla v9.4s, v20.4s, v1.s[1]\n"
    "fmla v10.4s, v20.4s, v1.s[2]\n"
    "fmla v11.4s, v20.4s, v1.s[3]\n"
    "fmla v12.4s, v21.4s, v0.s[0]\n"
    "fmla v13.4s, v21.4s, v0.s[1]\n"
    "fmla v14.4s, v21.4s, v0.s[2]\n"
    "fmla v15.4s, v21.4s, v0.s[3]\n"
    "fmla v16.4s, v21.4s, v1.s[0]\n"
    "fmla v17.4s, v21.4s, v1.s[1]\n"
    "fmla v18.4s, v21.4s, v1.s[2]\n"
    "fmla v19.4s, v21.4s, v1.s[3]\n"

    "ldr q0, [%[in], #64]\n"    // in_C0hw[0:3]
    "ldr q1, [%[in], #80]\n"    // in_C0hw[4:7]
    "ldr q20, [%[wgt], #64]\n"    // wgt_C0O0o[0:3]
    "ldr q21, [%[wgt], #80]\n"    // wgt_C0O1o[0:3]
    "fmla v4.4s, v22.4s, v2.s[0]\n"
    "fmla v5.4s, v22.4s, v2.s[1]\n"
    "fmla v6.4s, v22.4s, v2.s[2]\n"
    "fmla v7.4s, v22.4s, v2.s[3]\n"
    "fmla v8.4s, v22.4s, v3.s[0]\n"
    "fmla v9.4s, v22.4s, v3.s[1]\n"
    "fmla v10.4s, v22.4s, v3.s[2]\n"
    "fmla v11.4s, v22.4s, v3.s[3]\n"
    "fmla v12.4s, v23.4s, v2.s[0]\n"
    "fmla v13.4s, v23.4s, v2.s[1]\n"
    "fmla v14.4s, v23.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v2.s[3]\n"
    "fmla v16.4s, v23.4s, v3.s[0]\n"
    "fmla v17.4s, v23.4s, v3.s[1]\n"
    "fmla v18.4s, v23.4s, v3.s[2]\n"
    "fmla v19.4s, v23.4s, v3.s[3]\n"
    "add %[in], %[in], #64\n"
    "add %[wgt], %[wgt], #64\n"
    "subs x0, x0, #2\n"
    "bne 0b\n"
    "str q4, [%[out_0]]\n"        // out_O0HW0o[0:3]
    "str q5, [%[out_0], #16]\n"   // out_O0HW1o[0:3]
    "str q6, [%[out_0], #32]\n"   // out_O0HW2o[0:3]
    "str q7, [%[out_0], #48]\n"   // out_O0HW3o[0:3]
    "str q8, [%[out_0], #64]\n"   // out_O0HW4o[0:3]
    "str q9, [%[out_0], #80]\n"   // out_O0HW5o[0:3]
    "str q10, [%[out_0], #96]\n"  // out_O0HW6o[0:3]
    "str q11, [%[out_0], #112]\n" // out_O0HW7o[0:3]
    "str q12, [%[out_1]]\n"       // out_O1HW0o[0:3]
    "str q13, [%[out_1], #16]\n"  // out_O1HW1o[0:3]
    "str q14, [%[out_1], #32]\n"  // out_O1HW2o[0:3]
    "str q15, [%[out_1], #48]\n"  // out_O1HW3o[0:3]
    "str q16, [%[out_1], #64]\n"  // out_O1HW4o[0:3]
    "str q17, [%[out_1], #80]\n"  // out_O1HW5o[0:3]
    "str q18, [%[out_1], #96]\n"  // out_O1HW6o[0:3]
    "str q19, [%[out_1], #112]\n" // out_O1HW7o[0:3]
    :[out_0]"+r"(out_0),
     [out_1]"+r"(out_1),
     [in]"+r"(in),
     [wgt]"+r"(wgt)
    :[k]"r"(k),
     [b_0]"r"(b_0),
     [b_1]"r"(b_1)
    :"memory", "cc", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23", "q24", "q25", "x0"
  );
#else
  for (int c = 0; c < k; c++) {
    for (int hw8 = 0; hw8 < 8; hw8++) {
      for (int o4 = 0; o4 < 4; o4++) {
        if (c == 0) {
          out_0[hw8*4 + o4] = b_0[o4];
          out_1[hw8*4 + o4] = b_1[o4];
        }
        out_0[hw8*4 + o4] += in[c*8 + hw8] * wgt[c*8 + o4];
        out_1[hw8*4 + o4] += in[c*8 + hw8] * wgt[c*8 + 4 + o4];
      }
    }
  }
#endif
}


void sgemm_4x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k)
{
#if defined(_USE_NEON_A55)
  __asm__ __volatile__(
    "ldr q24, [%[b_0]]\n"         // b_O0o[0:3]
    "ldr q25, [%[b_1]]\n"         // b_O1o[0:3]
    "mov x0, %[k]\n"              // K (ic_blk)
    "ldr d0, [%[in]]\n"           // in_C0hw[0:1]
    "mov v4.16b, v24.16b\n"       // out_O0HW0o[0:3]
    "ldr x1, [%[in], #8]\n"       // in_C0hw[2:3]
    "mov v5.16b, v24.16b\n"       // out_O0HW1o[0:3]
    "ins v0.d[1], x1\n"           // in_C0hw[0:3]
    "mov v6.16b, v24.16b\n"       // out_O0HW2o[0:3]
    "ldr d20, [%[wgt]]\n"         // wgt_C0O0o[0:1]
    "mov v7.16b, v24.16b\n"       // out_O0HW3o[0:3]
    "ldr x3, [%[wgt], #8]\n"      // wgt_C0O0o[2:3]
    "mov v12.16b, v25.16b\n"      // out_O1HW0o[0:3]
    "ins v20.d[1], x3\n"          // wgt_C0O0o[0:3]
    "mov v13.16b, v25.16b\n"      // out_O1HW1o[0:3]
    "ldr d21, [%[wgt], #16]\n"    // wgt_C0O1o[0:1]
    "mov v14.16b, v25.16b\n"      // out_O1HW2o[0:3]
    "ldr x4, [%[wgt], #24]\n"     // wgt_C0O1o[2:3]
    "mov v15.16b, v25.16b\n"      // out_O1HW3o[0:3]
    "ins v21.d[1], x4\n"          // wgt_C0O1o[0:3]
    "0:\n"
    "ldr d2, [%[in], #16]\n"      // in_C1hw[0:1]
    "fmla v4.4s, v20.4s, v0.s[0]\n"
    "ldr x1, [%[in], #24]\n"      // in_C1hw[2:3]
    "fmla v5.4s, v20.4s, v0.s[1]\n"
    "ins v2.d[1], x1\n"           // in_C1hw[0:3]
    "fmla v6.4s, v20.4s, v0.s[2]\n"
    "ldr d22, [%[wgt], #32]\n"    // wgt_C1O0o[0:1]
    "fmla v7.4s, v20.4s, v0.s[3]\n"
    "ldr x3, [%[wgt], #40]\n"     // wgt_C1O0o[2:3]
    "fmla v12.4s, v21.4s, v0.s[0]\n"
    "ins v22.d[1], x3\n"          // wgt_C1O0o[0:3]
    "fmla v13.4s, v21.4s, v0.s[1]\n"
    "ldr d23, [%[wgt], #48]\n"    // wgt_C1O1o[0:1]
    "fmla v14.4s, v21.4s, v0.s[2]\n"
    "ldr x4, [%[wgt], #56]\n"     // wgt_C1O1o[2:3]
    "fmla v15.4s, v21.4s, v0.s[3]\n"
    "ins v23.d[1], x4\n"          // wgt_C1O1o[0:3]

    "ldr d0, [%[in], #32]\n"      // in_C0hw[0:1]
    "fmla v4.4s, v22.4s, v2.s[0]\n"
    "ldr x1, [%[in], #40]\n"      // in_C0hw[2:3]
    "fmla v5.4s, v22.4s, v2.s[1]\n"
    "ins v0.d[1], x1\n"           // in_C0hw[0:3]
    "fmla v6.4s, v22.4s, v2.s[2]\n"
    "ldr d20, [%[wgt], #64]\n"    // wgt_C0O0o[0:1]
    "fmla v7.4s, v22.4s, v2.s[3]\n"
    "ldr x3, [%[wgt], #72]\n"     // wgt_C0O0o[2:3]
    "fmla v12.4s, v23.4s, v2.s[0]\n"
    "ins v20.d[1], x3\n"          // wgt_C0O0o[0:3]
    "fmla v13.4s, v23.4s, v2.s[1]\n"
    "ldr d21, [%[wgt], #80]\n"    // wgt_C0O1o[0:1]
    "fmla v14.4s, v23.4s, v2.s[2]\n"
    "ldr x4, [%[wgt], #88]\n"     // wgt_C0O1o[2:3]
    "fmla v15.4s, v23.4s, v2.s[3]\n"
    "ins v21.d[1], x4\n"          // wgt_C0O1o[0:3]
    "add %[in], %[in], #32\n"
    "add %[wgt], %[wgt], #64\n"
    "subs x0, x0, #2\n"
    "bne 0b\n"
    "str q4, [%[out_0]]\n"        // out_O0HW0o[0:3]
    "str q5, [%[out_0], #16]\n"   // out_O0HW1o[0:3]
    "str q6, [%[out_0], #32]\n"   // out_O0HW2o[0:3]
    "str q7, [%[out_0], #48]\n"   // out_O0HW3o[0:3]
    "str q12, [%[out_1]]\n"       // out_O1HW0o[0:3]
    "str q13, [%[out_1], #16]\n"  // out_O1HW1o[0:3]
    "str q14, [%[out_1], #32]\n"  // out_O1HW2o[0:3]
    "str q15, [%[out_1], #48]\n"  // out_O1HW3o[0:3]
    :[out_0]"+r"(out_0),
     [out_1]"+r"(out_1),
     [in]"+r"(in),
     [wgt]"+r"(wgt)
    :[k]"r"(k),
     [b_0]"r"(b_0),
     [b_1]"r"(b_1)
    :"memory", "cc", "q0", "q2", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15", "q20", "q21", "q22", "q23", "q24", "q25", "x0", "x1", "x3", "x4"
  );
#elif defined(_USC_NEON_A76)
  __asm__ __volatile__(
    "ldr q24, [%[b_0]]\n"         // b_O0o[0:3]
    "ldr q25, [%[b_1]]\n"         // b_O1o[0:3]
    "mov x0, %[k]\n"              // K (ic_blk)
    "mov v4.16b, v24.16b\n"       // out_O0HW0o[0:3]
    "mov v5.16b, v24.16b\n"       // out_O0HW1o[0:3]
    "mov v6.16b, v24.16b\n"       // out_O0HW2o[0:3]
    "mov v7.16b, v24.16b\n"       // out_O0HW3o[0:3]
    "mov v12.16b, v25.16b\n"      // out_O1HW0o[0:3]
    "mov v13.16b, v25.16b\n"      // out_O1HW1o[0:3]
    "mov v14.16b, v25.16b\n"      // out_O1HW2o[0:3]
    "mov v15.16b, v25.16b\n"      // out_O1HW3o[0:3]
    "ldr q0, [%[in]]\n"           // in_C0hw[0:3]
    "ldr q20, [%[wgt]]\n"         // wgt_C0O0o[0:3]
    "ldr q21, [%[wgt], #16]\n"    // wgt_C0O1o[0:3]
    "0:\n"
    "ldr q2, [%[in], #16]\n"      // in_C1hw[0:3]
    "ldr q22, [%[wgt], #32]\n"    // wgt_C1O0o[0:3]
    "ldr q23, [%[wgt], #48]\n"    // wgt_C1O1o[0:3]
    "fmla v4.4s, v20.4s, v0.s[0]\n"
    "fmla v5.4s, v20.4s, v0.s[1]\n"
    "fmla v6.4s, v20.4s, v0.s[2]\n"
    "fmla v7.4s, v20.4s, v0.s[3]\n"
    "fmla v12.4s, v21.4s, v0.s[0]\n"
    "fmla v13.4s, v21.4s, v0.s[1]\n"
    "fmla v14.4s, v21.4s, v0.s[2]\n"
    "fmla v15.4s, v21.4s, v0.s[3]\n"

    "ldr q0, [%[in], #32]\n"    // in_C0hw[0:3]
    "ldr q20, [%[wgt], #64]\n"    // wgt_C0O0o[0:3]
    "ldr q21, [%[wgt], #80]\n"    // wgt_C0O1o[0:3]
    "fmla v4.4s, v22.4s, v2.s[0]\n"
    "fmla v5.4s, v22.4s, v2.s[1]\n"
    "fmla v6.4s, v22.4s, v2.s[2]\n"
    "fmla v7.4s, v22.4s, v2.s[3]\n"
    "fmla v12.4s, v23.4s, v2.s[0]\n"
    "fmla v13.4s, v23.4s, v2.s[1]\n"
    "fmla v14.4s, v23.4s, v2.s[2]\n"
    "fmla v15.4s, v23.4s, v2.s[3]\n"
    "add %[in], %[in], #32\n"
    "add %[wgt], %[wgt], #64\n"
    "subs x0, x0, #2\n"
    "bne 0b\n"
    "str q4, [%[out_0]]\n"        // out_O0HW0o[0:3]
    "str q5, [%[out_0], #16]\n"   // out_O0HW1o[0:3]
    "str q6, [%[out_0], #32]\n"   // out_O0HW2o[0:3]
    "str q7, [%[out_0], #48]\n"   // out_O0HW3o[0:3]
    "str q12, [%[out_1]]\n"       // out_O1HW0o[0:3]
    "str q13, [%[out_1], #16]\n"  // out_O1HW1o[0:3]
    "str q14, [%[out_1], #32]\n"  // out_O1HW2o[0:3]
    "str q15, [%[out_1], #48]\n"  // out_O1HW3o[0:3]
    :[out_0]"+r"(out_0),
     [out_1]"+r"(out_1),
     [in]"+r"(in),
     [wgt]"+r"(wgt)
    :[k]"r"(k),
     [b_0]"r"(b_0),
     [b_1]"r"(b_1)
    :"memory", "cc", "q0", "q2", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15", "q20", "q21", "q22", "q23", "q24", "q25", "x0"
  );
#else
  for (int c = 0; c < k; c++) {
    for (int hw4 = 0; hw4 < 4; hw4++) {
      for (int o4 = 0; o4 < 4; o4++) {
        if (c == 0) {
          out_0[hw4*4 + o4] = b_0[o4];
          out_1[hw4*4 + o4] = b_1[o4];
        }
        out_0[hw4*4 + o4] += in[c*4 + hw4] * wgt[c*8 + o4];
        out_1[hw4*4 + o4] += in[c*4 + hw4] * wgt[c*8 + 4 + o4];
      }
    }
  }
#endif
}


void sgemm_1x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k)
{
#if defined(_USE_NEON_A55)
  __asm__ __volatile__(
    "ldr q24, [%[b_0]]\n"         // b_O0o[0:3]
    "ldr q25, [%[b_1]]\n"         // b_O1o[0:3]
    "mov x0, %[k]\n"              // K (ic_blk)
    "ldr s0, [%[in]]\n"           // in_C0hw[0]
    "mov v4.16b, v24.16b\n"       // out_O0HW0o[0:3]
    "ldr d20, [%[wgt]]\n"         // wgt_C0O0o[0:1]
    "ldr x3, [%[wgt], #8]\n"      // wgt_C0O0o[2:3]
    "mov v12.16b, v25.16b\n"      // out_O1HW0o[0:3]
    "ins v20.d[1], x3\n"          // wgt_C0O0o[0:3]
    "ldr d21, [%[wgt], #16]\n"    // wgt_C0O1o[0:1]
    "ldr x4, [%[wgt], #24]\n"     // wgt_C0O1o[2:3]
    "ins v21.d[1], x4\n"          // wgt_C0O1o[0:3]
    "0:\n"
    "ldr s2, [%[in], #4]\n"      // in_C1hw[0:1]
    "fmla v4.4s, v20.4s, v0.s[0]\n"
    "ldr d22, [%[wgt], #32]\n"    // wgt_C1O0o[0:1]
    "ldr x3, [%[wgt], #40]\n"     // wgt_C1O0o[2:3]
    "fmla v12.4s, v21.4s, v0.s[0]\n"
    "ins v22.d[1], x3\n"          // wgt_C1O0o[0:3]
    "ldr d23, [%[wgt], #48]\n"    // wgt_C1O1o[0:1]
    "ldr x4, [%[wgt], #56]\n"     // wgt_C1O1o[2:3]
    "ins v23.d[1], x4\n"          // wgt_C1O1o[0:3]

    "ldr s0, [%[in], #8]\n"      // in_C0hw[0:1]
    "fmla v4.4s, v22.4s, v2.s[0]\n"
    "ldr d20, [%[wgt], #64]\n"    // wgt_C0O0o[0:1]
    "ldr x3, [%[wgt], #72]\n"     // wgt_C0O0o[2:3]
    "fmla v12.4s, v23.4s, v2.s[0]\n"
    "ins v20.d[1], x3\n"          // wgt_C0O0o[0:3]
    "ldr d21, [%[wgt], #80]\n"    // wgt_C0O1o[0:1]
    "ldr x4, [%[wgt], #88]\n"     // wgt_C0O1o[2:3]
    "ins v21.d[1], x4\n"          // wgt_C0O1o[0:3]
    "add %[in], %[in], #8\n"
    "add %[wgt], %[wgt], #64\n"
    "subs x0, x0, #2\n"
    "bne 0b\n"
    "str q4, [%[out_0]]\n"        // out_O0HW0o[0:3]
    "str q12, [%[out_1]]\n"       // out_O1HW0o[0:3]
    :[out_0]"+r"(out_0),
     [out_1]"+r"(out_1),
     [in]"+r"(in),
     [wgt]"+r"(wgt)
    :[k]"r"(k),
     [b_0]"r"(b_0),
     [b_1]"r"(b_1)
    :"memory", "cc", "q4",  "q12", "q20", "q21", "q22", "q23", "q24", "q25", "x0", "x3", "x4"
  );
#elif defined(_USC_NEON_A76)
  __asm__ __volatile__(
    "ldr q24, [%[b_0]]\n"         // b_O0o[0:3]
    "ldr q25, [%[b_1]]\n"         // b_O1o[0:3]
    "mov x0, %[k]\n"              // K (ic_blk)
    "mov v4.16b, v24.16b\n"       // out_O0HW0o[0:3]
    "mov v12.16b, v25.16b\n"      // out_O1HW0o[0:3]
    "ldr s0, [%[in]]\n"           // in_C0hw[0:3]
    "ldr q20, [%[wgt]]\n"         // wgt_C0O0o[0:3]
    "ldr q21, [%[wgt], #16]\n"    // wgt_C0O1o[0:3]
    "0:\n"
    "ldr s2, [%[in], #4]\n"      // in_C1hw[0:3]
    "ldr q22, [%[wgt], #32]\n"    // wgt_C1O0o[0:3]
    "ldr q23, [%[wgt], #48]\n"    // wgt_C1O1o[0:3]
    "fmla v4.4s, v20.4s, v0.s[0]\n"
    "fmla v12.4s, v21.4s, v0.s[0]\n"

    "ldr s0, [%[in], #8]\n"    // in_C0hw[0:3]
    "ldr q20, [%[wgt], #64]\n"    // wgt_C0O0o[0:3]
    "ldr q21, [%[wgt], #80]\n"    // wgt_C0O1o[0:3]
    "fmla v4.4s, v22.4s, v2.s[0]\n"
    "fmla v12.4s, v23.4s, v2.s[0]\n"
    "add %[in], %[in], #8\n"
    "add %[wgt], %[wgt], #64\n"
    "subs x0, x0, #2\n"
    "bne 0b\n"
    "str q4, [%[out_0]]\n"        // out_O0HW0o[0:3]
    "str q12, [%[out_1]]\n"       // out_O1HW0o[0:3]
    :[out_0]"+r"(out_0),
     [out_1]"+r"(out_1),
     [in]"+r"(in),
     [wgt]"+r"(wgt)
    :[k]"r"(k),
     [b_0]"r"(b_0),
     [b_1]"r"(b_1)
    :"memory", "cc", "q0", "q2", "q4", "q12", "q20", "q21", "q22", "q23", "q24", "q25", "x0"
  );
#else
  for (int c = 0; c < k; c++) {
    for (int hw1 = 0; hw1 < 1; hw1++) {
      for (int o4 = 0; o4 < 4; o4++) {
        if (c == 0) {
          out_0[hw1*4 + o4] = b_0[o4];
          out_1[hw1*4 + o4] = b_1[o4];
        }
        out_0[hw1*4 + o4] += in[c*1 + hw1] * wgt[c*8 + o4];
        out_1[hw1*4 + o4] += in[c*1 + hw1] * wgt[c*8 + 4 + o4];
      }
    }
  }
#endif
}


// NOT USED
void sgemm_8x4(float *in, float *wgt, float *out_0, float *b_0, int k)
{
  for (int c = 0; c < k; c++) {
    for (int hw8 = 0; hw8 < 8; hw8++) {
      for (int o4 = 0; o4 < 4; o4++) {
        if (c == 0) {
          out_0[hw8*4 + o4] = b_0[o4];
        }
        out_0[hw8*4 + o4] += in[c*8 + hw8] * wgt[c*4 + o4];
      }
    }
  }
}


void sgemm_4x4(float *in, float *wgt, float *out_0, float *b_0, int k)
{
  for (int c = 0; c < k; c++) {
    for (int hw4 = 0; hw4 < 4; hw4++) {
      for (int o4 = 0; o4 < 4; o4++) {
        if (c == 0) {
          out_0[hw4*4 + o4] = b_0[o4];
        }
        out_0[hw4*4 + o4] += in[c*4 + hw4] * wgt[c*4 + o4];
      }
    }
  }
}


void sgemm_1x4(float *in, float *wgt, float *out_0, float *b_0, int k)
{
  for (int c = 0; c < k; c++) {
    for (int hw1 = 0; hw1 < 1; hw1++) {
      for (int o4 = 0; o4 < 4; o4++) {
        if (c == 0) {
          out_0[hw1*4 + o4] = b_0[o4];
        }
        out_0[hw1*4 + o4] += in[c*1 + hw1] * wgt[c*4 + o4];
      }
    }
  }
}