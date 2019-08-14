#include "conv.hpp"
#include <iostream>
#include <stdlib.h>
#include <string.h>


// More description see header for detail.


void weight_trans_im2col(float *weight, float *weight_tm, int ic, int oc, int fh, int fw)
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


void conv_im2col_total_pack(float *input, float *weight, float *output, float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf)
{
  float* in_pad = buf;
  int ohow = oh*ow;
  int ih_pad = ih + 2*p;
  int iw_pad = iw + 2*p;
  ic /= 4;
  oc /= 4;
  float* in_pack = buf + ic*ih_pad*iw_pad*4;
  for (int n = 0; n < nb; n++) {
    // copy into buff as input_pad
    if (p == 0) {
      in_pad = (float*)input;
    } else {
      for(int c = 0; c < ic; c++) {
        float* in_pad_mov = in_pad + c*ih_pad*iw_pad*4;
        float* input_mov = input + c*ih*iw*4; 
        memset(in_pad_mov, 0, p*iw_pad*4*sizeof(float));
        for (int h = 0; h < ih; h++) {
          memset(in_pad_mov + p*iw_pad*4 + h*iw_pad*4, 0, p*4*sizeof(float));
          memcpy(in_pad_mov + p*iw_pad*4 + h*iw_pad*4 + p*4, input_mov + h*iw*4, iw*4*sizeof(float));
          memset(in_pad_mov + p*iw_pad*4 + h*iw_pad*4 + p*4 + iw*4, 0, p*4*sizeof(float));
        }
        memset(in_pad_mov + p*iw_pad*4 + ih*iw_pad*4, 0, p*iw_pad*4*sizeof(float));
      }
    }

    // handle 2|oc
    for (int o = 0; o < oc-1; o+=2) {
      float* out_0 = output + n*oc*ohow*4 + o*ohow*4;
      float* out_1 = out_0 + ohow*4;
      float* b_0 = bias + o*4;
      float* b_1 = b_0 + 4;
      float* wgt_0 = weight + o*4*fh*fw*ic*4;

      // handle 8|hw
      for (int hw = 0; hw < ohow-7; hw+=8) {
        float *in_pack_hw = in_pack + hw*fh*fw*ic*4;
        if (o == 0) {
          // input packing
          int h0 = (hw/ow)*s;
          int w0 = (hw%ow)*s;
          int h1 = ((hw+1)/ow)*s;
          int w1 = ((hw+1)%ow)*s;
          int h2 = ((hw+2)/ow)*s;
          int w2 = ((hw+2)%ow)*s;
          int h3 = ((hw+3)/ow)*s;
          int w3 = ((hw+3)%ow)*s;
          int h4 = ((hw+4)/ow)*s;
          int w4 = ((hw+4)%ow)*s;
          int h5 = ((hw+5)/ow)*s;
          int w5 = ((hw+5)%ow)*s;
          int h6 = ((hw+6)/ow)*s;
          int w6 = ((hw+6)%ow)*s;
          int h7 = ((hw+7)/ow)*s;
          int w7 = ((hw+7)%ow)*s;
          for (int c = 0; c < ic; c++) {
            for (int fh_idx = 0; fh_idx < fh; fh_idx++) {
              for (int fw_idx = 0; fw_idx < fw; fw_idx++) {
                float* in_mov = in_pad + n*ic*ih_pad*iw_pad*4 + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
                float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*8 + fw_idx*ic*4*8 + c*4*8;
                float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
                float *in_1 = in_mov + h1*iw_pad*4 + w1*4;
                float *in_2 = in_mov + h2*iw_pad*4 + w2*4;
                float *in_3 = in_mov + h3*iw_pad*4 + w3*4;
                float *in_4 = in_mov + h4*iw_pad*4 + w4*4;
                float *in_5 = in_mov + h5*iw_pad*4 + w5*4;
                float *in_6 = in_mov + h6*iw_pad*4 + w6*4;
                float *in_7 = in_mov + h7*iw_pad*4 + w7*4;
                //TODO use neon
                for (int c4 = 0; c4 < 4; c4++) {
                  in_pack_fhfwic[c4*8] = in_0[c4];
                  in_pack_fhfwic[c4*8 + 1] = in_1[c4];
                  in_pack_fhfwic[c4*8 + 2] = in_2[c4];
                  in_pack_fhfwic[c4*8 + 3] = in_3[c4];
                  in_pack_fhfwic[c4*8 + 4] = in_4[c4];
                  in_pack_fhfwic[c4*8 + 5] = in_5[c4];
                  in_pack_fhfwic[c4*8 + 6] = in_6[c4];
                  in_pack_fhfwic[c4*8 + 7] = in_7[c4];
                }
              }
            }
          }
        }
        //TODO use neon
        for (int c = 0; c < fh*fw*ic*4; c++) {
          for (int hw8 = 0; hw8 < 8; hw8++) {
            for (int o4 = 0; o4 < 4; o4++) {
              if (c == 0) {
                out_0[hw8*4 + o4] = b_0[o4];
                out_1[hw8*4 + o4] = b_1[o4];
              }
              out_0[hw8*4 + o4] += in_pack_hw[c*8 + hw8] * wgt_0[c*8 + o4];
              out_1[hw8*4 + o4] += in_pack_hw[c*8 + hw8] * wgt_0[c*8 + 4 + o4];
            }
          }
        }
        out_0 += 8*4;
        out_1 += 8*4;
      }

      // handle 4|(hw%8)

      // handle hw%4
    }

    // handle oc%2
  }
}


void conv_im2col_tile_pack(float *input, float *weight, float *output, float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf)
{
  float* in_pad = buf;
  int ohow = oh*ow;
  int ih_pad = ih + 2*p;
  int iw_pad = iw + 2*p;
  ic /= 4;
  oc /= 4;
  float* in_pack = buf + ic*ih_pad*iw_pad*4;
  for (int n = 0; n < nb; n++) {
    // copy into buff as input_pad
    if (p == 0) {
      in_pad = (float*)input;
    } else {
      for(int c = 0; c < ic; c++) {
        float* in_pad_mov = in_pad + c*ih_pad*iw_pad*4;
        float* input_mov = input + c*ih*iw*4; 
        memset(in_pad_mov, 0, p*iw_pad*4*sizeof(float));
        for (int h = 0; h < ih; h++) {
          memset(in_pad_mov + p*iw_pad*4 + h*iw_pad*4, 0, p*4*sizeof(float));
          memcpy(in_pad_mov + p*iw_pad*4 + h*iw_pad*4 + p*4, input_mov + h*iw*4, iw*4*sizeof(float));
          memset(in_pad_mov + p*iw_pad*4 + h*iw_pad*4 + p*4 + iw*4, 0, p*4*sizeof(float));
        }
        memset(in_pad_mov + p*iw_pad*4 + ih*iw_pad*4, 0, p*iw_pad*4*sizeof(float));
      }
    }

    // handle 8|hw
    for (int hw = 0; hw < ohow-7; hw+=8) {
      float *in_pack_hw = in_pack;
      // input packings
      int h0 = (hw/ow)*s;
      int w0 = (hw%ow)*s;
      int h1 = ((hw+1)/ow)*s;
      int w1 = ((hw+1)%ow)*s;
      int h2 = ((hw+2)/ow)*s;
      int w2 = ((hw+2)%ow)*s;
      int h3 = ((hw+3)/ow)*s;
      int w3 = ((hw+3)%ow)*s;
      int h4 = ((hw+4)/ow)*s;
      int w4 = ((hw+4)%ow)*s;
      int h5 = ((hw+5)/ow)*s;
      int w5 = ((hw+5)%ow)*s;
      int h6 = ((hw+6)/ow)*s;
      int w6 = ((hw+6)%ow)*s;
      int h7 = ((hw+7)/ow)*s;
      int w7 = ((hw+7)%ow)*s;
      for (int c = 0; c < ic; c++) {
        for (int fh_idx = 0; fh_idx < fh; fh_idx++) {
          for (int fw_idx = 0; fw_idx < fw; fw_idx++) {
            float* in_mov = in_pad + n*ic*ih_pad*iw_pad*4 + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
            float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*8 + fw_idx*ic*4*8 + c*4*8;
            float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
            float *in_1 = in_mov + h1*iw_pad*4 + w1*4;
            float *in_2 = in_mov + h2*iw_pad*4 + w2*4;
            float *in_3 = in_mov + h3*iw_pad*4 + w3*4;
            float *in_4 = in_mov + h4*iw_pad*4 + w4*4;
            float *in_5 = in_mov + h5*iw_pad*4 + w5*4;
            float *in_6 = in_mov + h6*iw_pad*4 + w6*4;
            float *in_7 = in_mov + h7*iw_pad*4 + w7*4;
            //TODO use neon
            for (int c4 = 0; c4 < 4; c4++) {
              in_pack_fhfwic[c4*8] = in_0[c4];
              in_pack_fhfwic[c4*8 + 1] = in_1[c4];
              in_pack_fhfwic[c4*8 + 2] = in_2[c4];
              in_pack_fhfwic[c4*8 + 3] = in_3[c4];
              in_pack_fhfwic[c4*8 + 4] = in_4[c4];
              in_pack_fhfwic[c4*8 + 5] = in_5[c4];
              in_pack_fhfwic[c4*8 + 6] = in_6[c4];
              in_pack_fhfwic[c4*8 + 7] = in_7[c4];
            }
          }
        }
      }

      // handle 2|oc
      for (int o = 0; o < oc-1; o+=2) {
        float* out_0 = output + n*oc*ohow*4 + o*ohow*4 + hw*4;
        float* out_1 = out_0 + ohow*4;
        float* b_0 = bias + o*4;
        float* b_1 = b_0 + 4;
        float* wgt_0 = weight + o*4*fh*fw*ic*4;

        //TODO use neon
        for (int c = 0; c < fh*fw*ic*4; c++) {
          for (int hw8 = 0; hw8 < 8; hw8++) {
            for (int o4 = 0; o4 < 4; o4++) {
              if (c == 0) {
                out_0[hw8*4 + o4] = b_0[o4];
                out_1[hw8*4 + o4] = b_1[o4];
              }
              out_0[hw8*4 + o4] += in_pack_hw[c*8 + hw8] * wgt_0[c*8 + o4];
              out_1[hw8*4 + o4] += in_pack_hw[c*8 + hw8] * wgt_0[c*8 + 4 + o4];
            }
          }
        }

        // handle oc%2
      }
    }

    // handle 4|(hw%8)

    // handle hw%4
  }
}