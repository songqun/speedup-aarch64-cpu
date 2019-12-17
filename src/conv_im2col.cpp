#include "conv.hpp"
#include "sgemm.hpp"
#include <stdlib.h>
#include <string.h>

#if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
#include <arm_neon.h>
#endif


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

    for (int o = 0; o < oc; o+=2) {
      float* out_0 = output + n*oc*ohow*4 + o*ohow*4;
      float* out_1 = out_0 + ohow*4;
      float* b_0 = bias + o*4;
      float* b_1 = b_0 + 4;
      float* wgt = weight + o*4*fh*fw*ic*4;

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
                float* in_mov = in_pad + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
                float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*8 + fw_idx*ic*4*8 + c*4*8;
                float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
                float *in_1 = in_mov + h1*iw_pad*4 + w1*4;
                float *in_2 = in_mov + h2*iw_pad*4 + w2*4;
                float *in_3 = in_mov + h3*iw_pad*4 + w3*4;
                float *in_4 = in_mov + h4*iw_pad*4 + w4*4;
                float *in_5 = in_mov + h5*iw_pad*4 + w5*4;
                float *in_6 = in_mov + h6*iw_pad*4 + w6*4;
                float *in_7 = in_mov + h7*iw_pad*4 + w7*4;
        #if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
                float32x4_t v0 = vld1q_f32(in_0);
                float32x4_t v1 = vld1q_f32(in_1);
                float32x4_t v2 = vld1q_f32(in_2);
                float32x4_t v3 = vld1q_f32(in_3);
                float32x4_t v4 = vld1q_f32(in_4);
                float32x4_t v5 = vld1q_f32(in_5);
                float32x4_t v6 = vld1q_f32(in_6);
                float32x4_t v7 = vld1q_f32(in_7);
                __builtin_prefetch(in_0 + 64, 0, 3);
                __builtin_prefetch(in_1 + 64, 0, 3);
                __builtin_prefetch(in_2 + 64, 0, 3);
                __builtin_prefetch(in_3 + 64, 0, 3);
                __builtin_prefetch(in_4 + 64, 0, 3);
                __builtin_prefetch(in_5 + 64, 0, 3);
                __builtin_prefetch(in_6 + 64, 0, 3);
                __builtin_prefetch(in_7 + 64, 0, 3);
                vst1q_f32(in_pack_fhfwic,
                    vzip1q_f32(
                        vzip1q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                        vzip1q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
                vst1q_f32(in_pack_fhfwic + 4,
                    vzip2q_f32(
                        vzip1q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                        vzip1q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
                vst1q_f32(in_pack_fhfwic + 4*2,
                    vzip1q_f32(
                        vzip2q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                        vzip2q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
                vst1q_f32(in_pack_fhfwic + 4*3,
                    vzip2q_f32(
                        vzip2q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                        vzip2q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
                vst1q_f32(in_pack_fhfwic + 4*4,
                    vzip1q_f32(
                        vzip1q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                        vzip1q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
                vst1q_f32(in_pack_fhfwic + 4*5,
                    vzip2q_f32(
                        vzip1q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                        vzip1q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
                vst1q_f32(in_pack_fhfwic + 4*6,
                    vzip1q_f32(
                        vzip2q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                        vzip2q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
                vst1q_f32(in_pack_fhfwic + 4*7,
                    vzip2q_f32(
                        vzip2q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                        vzip2q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
        #else
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
        #endif
              }
            }
          }
        }
        sgemm_8x8(in_pack_hw, wgt, out_0, out_1, b_0, b_1, fh*fw*ic*4);
        out_0 += 8*4;
        out_1 += 8*4;
      }

      // handle 4|(hw%8)
      int hw_start = (ohow/8)*8;
      for (int hw = hw_start; hw < ohow-3; hw+=4) {
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
          for (int c = 0; c < ic; c++) {
            for (int fh_idx = 0; fh_idx < fh; fh_idx++) {
              for (int fw_idx = 0; fw_idx < fw; fw_idx++) {
                float* in_mov = in_pad + n*ic*ih_pad*iw_pad*4 + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
                float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*4 + fw_idx*ic*4*4 + c*4*4;
                float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
                float *in_1 = in_mov + h1*iw_pad*4 + w1*4;
                float *in_2 = in_mov + h2*iw_pad*4 + w2*4;
                float *in_3 = in_mov + h3*iw_pad*4 + w3*4;
        #if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
                __asm__ __volatile__(
                  "ldr q0, [%[in_0]]\n"
                  "prfm pldl1keep, [%[in_0], #256]\n"
                  "ldr q1, [%[in_1]]\n"
                  "prfm pldl1keep, [%[in_1], #256]\n"
                  "ldr q2, [%[in_2]]\n"
                  "prfm pldl1keep, [%[in_2], #256]\n"
                  "ldr q3, [%[in_3]]\n"
                  "prfm pldl1keep, [%[in_3], #256]\n"
                  "st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[in_pack_fhfwic]]\n"
                  :[in_pack_fhfwic]"+r"(in_pack_fhfwic)
                  :[in_0]"r"(in_0),
                   [in_1]"r"(in_1),
                   [in_2]"r"(in_2),
                   [in_3]"r"(in_3)
                  :"memory", "cc", "q0", "q1", "q2", "q3"
                );
        #else
                for (int c4 = 0; c4 < 4; c4++) {
                  in_pack_fhfwic[c4*4] = in_0[c4];
                  in_pack_fhfwic[c4*4 + 1] = in_1[c4];
                  in_pack_fhfwic[c4*4 + 2] = in_2[c4];
                  in_pack_fhfwic[c4*4 + 3] = in_3[c4];
                }
        #endif
              }
            }
          }
        }
        sgemm_4x8(in_pack_hw, wgt, out_0, out_1, b_0, b_1, fh*fw*ic*4);
        out_0 += 4*4;
        out_1 += 4*4;
      }

      // handle hw%4
      hw_start = (ohow/4)*4;
      for (int hw = hw_start; hw < ohow; hw++) {
        float *in_pack_hw = in_pack + hw*fh*fw*ic*4;
        if (o == 0) {
          // input packing
          int h0 = (hw/ow)*s;
          int w0 = (hw%ow)*s;
          for (int c = 0; c < ic; c++) {
            for (int fh_idx = 0; fh_idx < fh; fh_idx++) {
              for (int fw_idx = 0; fw_idx < fw; fw_idx++) {
                float* in_mov = in_pad + n*ic*ih_pad*iw_pad*4 + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
                float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*1 + fw_idx*ic*4*1 + c*4*1;
                float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
                //
                // for (int c4 = 0; c4 < 4; c4++) {
                //   in_pack_fhfwic[c4] = in_0[c4];
                // }
                //
                memcpy(in_pack_fhfwic, in_0, 4*sizeof(float));
                __builtin_prefetch(in_0 + 64, 0, 3);
              }
            }
          }
        }
        sgemm_1x8(in_pack_hw, wgt, out_0, out_1, b_0, b_1, fh*fw*ic*4);
        out_0 += 1*4;
        out_1 += 1*4;
      }
    }
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

      // input packing
      {
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
              float* in_mov = in_pad + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
              float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*8 + fw_idx*ic*4*8 + c*4*8;
              float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
              float *in_1 = in_mov + h1*iw_pad*4 + w1*4;
              float *in_2 = in_mov + h2*iw_pad*4 + w2*4;
              float *in_3 = in_mov + h3*iw_pad*4 + w3*4;
              float *in_4 = in_mov + h4*iw_pad*4 + w4*4;
              float *in_5 = in_mov + h5*iw_pad*4 + w5*4;
              float *in_6 = in_mov + h6*iw_pad*4 + w6*4;
              float *in_7 = in_mov + h7*iw_pad*4 + w7*4;
      #if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
              float32x4_t v0 = vld1q_f32(in_0);
              float32x4_t v1 = vld1q_f32(in_1);
              float32x4_t v2 = vld1q_f32(in_2);
              float32x4_t v3 = vld1q_f32(in_3);
              float32x4_t v4 = vld1q_f32(in_4);
              float32x4_t v5 = vld1q_f32(in_5);
              float32x4_t v6 = vld1q_f32(in_6);
              float32x4_t v7 = vld1q_f32(in_7);
              __builtin_prefetch(in_0 + 64, 0, 3);
              __builtin_prefetch(in_1 + 64, 0, 3);
              __builtin_prefetch(in_2 + 64, 0, 3);
              __builtin_prefetch(in_3 + 64, 0, 3);
              __builtin_prefetch(in_4 + 64, 0, 3);
              __builtin_prefetch(in_5 + 64, 0, 3);
              __builtin_prefetch(in_6 + 64, 0, 3);
              __builtin_prefetch(in_7 + 64, 0, 3);
              vst1q_f32(in_pack_fhfwic,
                  vzip1q_f32(
                      vzip1q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                      vzip1q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
              vst1q_f32(in_pack_fhfwic + 4,
                  vzip2q_f32(
                      vzip1q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                      vzip1q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
              vst1q_f32(in_pack_fhfwic + 4*2,
                  vzip1q_f32(
                      vzip2q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                      vzip2q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
              vst1q_f32(in_pack_fhfwic + 4*3,
                  vzip2q_f32(
                      vzip2q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                      vzip2q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
              vst1q_f32(in_pack_fhfwic + 4*4,
                  vzip1q_f32(
                      vzip1q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                      vzip1q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
              vst1q_f32(in_pack_fhfwic + 4*5,
                  vzip2q_f32(
                      vzip1q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                      vzip1q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
              vst1q_f32(in_pack_fhfwic + 4*6,
                  vzip1q_f32(
                      vzip2q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                      vzip2q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
              vst1q_f32(in_pack_fhfwic + 4*7,
                  vzip2q_f32(
                      vzip2q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                      vzip2q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
      #else
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
      #endif
            }
          }
        }
      }

      for (int o = 0; o < oc; o+=2) {
        float* out_0 = output + n*oc*ohow*4 + o*ohow*4 + hw*4;
        float* out_1 = out_0 + ohow*4;
        float* b_0 = bias + o*4;
        float* b_1 = b_0 + 4;
        float* wgt = weight + o*4*fh*fw*ic*4;

        sgemm_8x8(in_pack_hw, wgt, out_0, out_1, b_0, b_1, fh*fw*ic*4);
      }
    }

    // handle 4|(hw%8)
    int hw_start = (ohow/8)*8;
    for (int hw = hw_start; hw < ohow-3; hw+=4) {
      float *in_pack_hw = in_pack;
      // input packing
      {
        int h0 = (hw/ow)*s;
        int w0 = (hw%ow)*s;
        int h1 = ((hw+1)/ow)*s;
        int w1 = ((hw+1)%ow)*s;
        int h2 = ((hw+2)/ow)*s;
        int w2 = ((hw+2)%ow)*s;
        int h3 = ((hw+3)/ow)*s;
        int w3 = ((hw+3)%ow)*s;
        for (int c = 0; c < ic; c++) {
          for (int fh_idx = 0; fh_idx < fh; fh_idx++) {
            for (int fw_idx = 0; fw_idx < fw; fw_idx++) {
              float* in_mov = in_pad + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
              float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*4 + fw_idx*ic*4*4 + c*4*4;
              float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
              float *in_1 = in_mov + h1*iw_pad*4 + w1*4;
              float *in_2 = in_mov + h2*iw_pad*4 + w2*4;
              float *in_3 = in_mov + h3*iw_pad*4 + w3*4;
      #if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
              __asm__ __volatile__(
                "ldr q0, [%[in_0]]\n"
                "prfm pldl1keep, [%[in_0], #256]\n"
                "ldr q1, [%[in_1]]\n"
                "prfm pldl1keep, [%[in_1], #256]\n"
                "ldr q2, [%[in_2]]\n"
                "prfm pldl1keep, [%[in_2], #256]\n"
                "ldr q3, [%[in_3]]\n"
                "prfm pldl1keep, [%[in_3], #256]\n"
                "st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[in_pack_fhfwic]]\n"
                :[in_pack_fhfwic]"+r"(in_pack_fhfwic)
                :[in_0]"r"(in_0),
                  [in_1]"r"(in_1),
                  [in_2]"r"(in_2),
                  [in_3]"r"(in_3)
                :"memory", "cc", "q0", "q1", "q2", "q3"
              );
      #else
              for (int c4 = 0; c4 < 4; c4++) {
                in_pack_fhfwic[c4*4] = in_0[c4];
                in_pack_fhfwic[c4*4 + 1] = in_1[c4];
                in_pack_fhfwic[c4*4 + 2] = in_2[c4];
                in_pack_fhfwic[c4*4 + 3] = in_3[c4];
              }
      #endif
            }
          }
        }
      }

      for (int o = 0; o < oc; o+=2) {
        float* out_0 = output + n*oc*ohow*4 + o*ohow*4 + hw*4;
        float* out_1 = out_0 + ohow*4;
        float* b_0 = bias + o*4;
        float* b_1 = b_0 + 4;
        float* wgt = weight + o*4*fh*fw*ic*4;

        sgemm_4x8(in_pack_hw, wgt, out_0, out_1, b_0, b_1, fh*fw*ic*4);
      }
    }

    // handle hw%4
    hw_start = (ohow/4)*4;
    for (int hw = hw_start; hw < ohow; hw++) {
      float *in_pack_hw = in_pack;
      // input packing
      {
        int h0 = (hw/ow)*s;
        int w0 = (hw%ow)*s;
        for (int c = 0; c < ic; c++) {
          for (int fh_idx = 0; fh_idx < fh; fh_idx++) {
            for (int fw_idx = 0; fw_idx < fw; fw_idx++) {
              float* in_mov = in_pad + c*ih_pad*iw_pad*4 + fh_idx*iw_pad*4 + fw_idx*4;
              float* in_pack_fhfwic = in_pack_hw + fh_idx*fw*ic*4*1 + fw_idx*ic*4*1 + c*4*1;
              float *in_0 = in_mov + h0*iw_pad*4 + w0*4;
              //
              // for (int c4 = 0; c4 < 4; c4++) {
              //   in_pack_fhfwic[c4] = in_0[c4];
              // }
              //
              memcpy(in_pack_fhfwic, in_0, 4*sizeof(float));
              __builtin_prefetch(in_0 + 64, 0, 3);
            }
          }
        }
      }

      for (int o = 0; o < oc; o+=2) {
        float* out_0 = output + n*oc*ohow*4 + o*ohow*4 + hw*4;
        float* out_1 = out_0 + ohow*4;
        float* b_0 = bias + o*4;
        float* b_1 = b_0 + 4;
        float* wgt = weight + o*4*fh*fw*ic*4;

        sgemm_1x8(in_pack_hw, wgt, out_0, out_1, b_0, b_1, fh*fw*ic*4);
      }
    }
  }
}