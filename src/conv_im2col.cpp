#include "conv.hpp"
#include <iostream>


// More description see header for detail.


void weight_trans_1x1s1p0(const float *weight, float *weight_tm, int ic, int oc, int fh, int fw)
{
  for (int o = 0; o < oc/8; o++) {
    for (int c = 0; c < ic; c++) {
      for (int o8 = 0; o8 < 8; o8++) {
        weight_tm[o*ic*8 + c*8 + o8] = weight[(o*8+o8)*ic + c];
      }
    }
  }
}


void weight_trans_im2col(const float *weight, float *weight_tm, int ic, int oc, int fh, int fw)
{
  for (int o = 0; o < oc/8; o++) {
    for (int hw = 0; hw < fh*fw; hw++) {
      for (int c = 0; c < ic; c++) {
        for (int o8 = 0; o8 < 8; o8++) {
          weight_tm[o*fh*fw*ic*8 + hw*ic*8 + c*8 + o8] = weight[(o*8+o8)*ic*fh*fw + c*fh*fw + hw];
        }
      }
    }
  }
}


void conv1x1s1p0(const float* input, const float* weight, float *output, const float* bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf)
{
  float* in_pack = buf;
  int ohow = oh*ow;
  for (int n = 0; n < nb; n++) {
    for (int o = 0; o < oc; o+=2) {
      float* out_0 = output + n*oc*ohow*4 + o*ohow*4;
      float* out_1 = out_0 + ohow*4;
      const float* b_0 = bias + o*4;
      const float* b_1 = b_0 + 4;
      const float* wgt_0 = weight + o*4*ic*4;

      for (int hw = 0; hw < ohow; hw+=8) {
        float *in_pack_0 = in_pack + hw*ic*4;
        if (o == 0) {
          // input packing
          const float* in_0 = input + n*ic*ih*iw*4;
          //TODO use neon
          for (int c = 0; c < ic; c++) {
            for (int hw8 = 0; hw8 < 8; hw8++) {
              for (int c4 = 0; c4 < 4; c4++) {
                in_pack_0[(c*4+c4)*8 + hw8] = in_0[c*ih*iw*4 + (hw+hw8)*4 + c4];
              }
            }
          }
        }
        //TODO use neon
        for (int c = 0; c < ic*4; c++) {
          for (int hw8 = 0; hw8 < 8; hw8++) {
            for (int o4 = 0; o4 < 4; o4++) {
              if (c == 0) {
                out_0[hw8*4 + o4] = b_0[o4];
                out_1[hw8*4 + o4] = b_1[o4];
              }
              out_0[hw8*4 + o4] += in_pack_0[c*8 + hw8] * wgt_0[c*8 + o4];
              out_1[hw8*4 + o4] += in_pack_0[c*8 + hw8] * wgt_0[c*8 + 4 + o4];
            }
          }
        }
        out_0 += 8*4;
        out_1 += 8*4;
      }
    }
  }
}


void conv_im2col(const float *input, const float *weight, float *output, const float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf)
{

}
