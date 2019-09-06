#include "sgemm.hpp"


void sgemm_8x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k)
{
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
}


void sgemm_4x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k)
{
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
}


void sgemm_1x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k)
{
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
}


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