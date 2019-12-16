#include <iostream>
#include "util.hpp"
#include <string.h>

#define FMLA \
        "fmla v0.4s, v1.4s, v2.s[0]\n" \
        "fmla v3.4s, v1.4s, v2.s[0]\n" \
        "fmla v4.4s, v1.4s, v2.s[0]\n" \
        "fmla v5.4s, v1.4s, v2.s[0]\n"

#define FMLA_LDR \
        "ldr d6, [%[a]]\n" \
        "fmla v0.4s, v1.4s, v2.s[0]\n" \
        "ldr d7, [%[a]]\n" \
        "fmla v3.4s, v1.4s, v2.s[0]\n" \
        "ldr d8, [%[a]]\n" \
        "fmla v4.4s, v1.4s, v2.s[0]\n" \
        "ldr d9, [%[a]]\n" \
        "fmla v5.4s, v1.4s, v2.s[0]\n"

#define KERNEL \
        "ldr d2, [%[a], #32]\n" \
        "fmla v4.4s, v20.4s, v0.s[0]\n" \
        "ldr x1, [%[a], #40]\n" \
        "fmla v5.4s, v20.4s, v0.s[1]\n" \
        "ins v2.d[1], x1\n" \
        "fmla v6.4s, v20.4s, v0.s[2]\n" \
        "ldr d3, [%[a], #48]\n" \
        "fmla v7.4s, v20.4s, v0.s[3]\n" \
        "ldr x2, [%[a], #56]\n" \
        "fmla v8.4s, v20.4s, v1.s[0]\n" \
        "ins v3.d[1], x2\n" \
        "fmla v9.4s, v20.4s, v1.s[1]\n" \
        "ldr d22, [%[a], #32]\n" \
        "fmla v10.4s, v20.4s, v1.s[2]\n" \
        "ldr x3, [%[a], #40]\n" \
        "fmla v11.4s, v20.4s, v1.s[3]\n" \
        "ins v22.d[1], x3\n" \
        "fmla v12.4s, v21.4s, v0.s[0]\n" \
        "ldr d23, [%[a], #48]\n" \
        "fmla v13.4s, v21.4s, v0.s[1]\n" \
        "ldr x4, [%[a], #56]\n" \
        "fmla v14.4s, v21.4s, v0.s[2]\n" \
        "ins v23.d[1], x4\n" \
        "fmla v15.4s, v21.4s, v0.s[3]\n" \
        "fmla v16.4s, v21.4s, v1.s[0]\n" \
        "fmla v17.4s, v21.4s, v1.s[1]\n" \
        "fmla v18.4s, v21.4s, v1.s[2]\n" \
        "fmla v19.4s, v21.4s, v1.s[3]\n" \
        "ldr d0, [%[a], #64]\n" \
        "fmla v4.4s, v22.4s, v2.s[0]\n" \
        "ldr x1, [%[a], #72]\n" \
        "fmla v5.4s, v22.4s, v2.s[1]\n" \
        "ins v0.d[1], x1\n" \
        "fmla v6.4s, v22.4s, v2.s[2]\n" \
        "ldr d1, [%[a], #80]\n" \
        "fmla v7.4s, v22.4s, v2.s[3]\n" \
        "ldr x2, [%[a], #88]\n" \
        "fmla v8.4s, v22.4s, v3.s[0]\n" \
        "ins v1.d[1], x2\n" \
        "fmla v9.4s, v22.4s, v3.s[1]\n" \
        "ldr d20, [%[a], #64]\n" \
        "fmla v10.4s, v22.4s, v3.s[2]\n" \
        "ldr x3, [%[a], #72]\n" \
        "fmla v11.4s, v22.4s, v3.s[3]\n" \
        "ins v20.d[1], x3\n" \
        "fmla v12.4s, v23.4s, v2.s[0]\n" \
        "ldr d21, [%[a], #80]\n" \
        "fmla v13.4s, v23.4s, v2.s[1]\n" \
        "ldr x4, [%[a], #88]\n" \
        "fmla v14.4s, v23.4s, v2.s[2]\n" \
        "ins v21.d[1], x4\n" \
        "fmla v15.4s, v23.4s, v2.s[3]\n" \
        "fmla v16.4s, v23.4s, v3.s[0]\n" \
        "fmla v17.4s, v23.4s, v3.s[1]\n" \
        "fmla v18.4s, v23.4s, v3.s[2]\n" \
        "fmla v19.4s, v23.4s, v3.s[3]\n" \


#define HHH KERNEL

#define TEST \
        HHH HHH HHH HHH \
        HHH HHH HHH HHH \
        HHH HHH HHH HHH \
        HHH HHH HHH HHH


int main() {
  float *a = (float*)malloc(1024*sizeof(float));
  float *b = (float*)malloc(1024*sizeof(float));
  int k = 100000;
  double start, end;
  start = get_current_time();
  __asm__ __volatile__(
    "mov x0, %[k]\n"
    "0:\n"
    TEST
    "subs x0, x0, #1\n"
    "bne 0b\n"
    :[b]"+r"(b)
    :[k]"r"(k),
     [a]"r"(a)
    :"memory", "cc", "x0", "x1", "x2", "x3", "x4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23", "q24", "q25", "q26", "q27", "q28", "q29"
  );
  end = get_current_time();
  std::cerr << (end-start)*1660000/16/k << "\n";
  free(a);
  free(b);
  return 0;
}