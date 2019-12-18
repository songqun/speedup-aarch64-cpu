#include "conv.hpp"
#include "sgemm.hpp"
#include <stdlib.h>
#include <string.h>
#include <iostream>


// More description see header for detail.


#if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
#include <arm_neon.h>


void trans_W_4x4_3x3(float* WTM[36], float* W[9])
{
  float T[6][3][8];

  float32x4_t v_01666 = vmovq_n_f32(0.1666666666666667f);
  float32x4_t v_minus_01666 = vmovq_n_f32(-0.1666666666666667f);
  float32x4_t v_00833 = vmovq_n_f32(0.0833333333333333f);
  float32x4_t v_minus_00833 = vmovq_n_f32(-0.0833333333333333f);
  float32x4_t v_004166 = vmovq_n_f32(0.0416666666666667f);
  float32x4_t v_025 = vmovq_n_f32(0.25f);

  for (int i = 0; i < 3; i++) {
      float32x4_t v_W0 = vld1q_f32(W[0*3+i]);
      float32x4_t v_W1 = vld1q_f32(W[1*3+i]);
      float32x4_t v_W2 = vld1q_f32(W[2*3+i]);

      float32x4_t v_t0 = vmulq_f32(v_01666, v_W2);
      float32x4_t v_t1 = vsubq_f32(vmulq_f32(v_minus_01666, v_W0), v_t0);
      float32x4_t v_t2 = vfmaq_f32(v_t0, v_004166, v_W0);

      float32x4_t v_T0 = vmulq_f32(v_025, v_W0);
      float32x4_t v_T1 = vfmaq_f32(v_t1, v_minus_01666, v_W1);
      float32x4_t v_T2 = vfmaq_f32(v_t1, v_01666, v_W1);
      float32x4_t v_T3 = vfmaq_f32(v_t2, v_00833, v_W1);
      float32x4_t v_T4 = vfmaq_f32(v_t2, v_minus_00833, v_W1);

      vst1q_f32(T[0][i], v_T0);
      vst1q_f32(T[1][i], v_T1);
      vst1q_f32(T[2][i], v_T2);
      vst1q_f32(T[3][i], v_T3);
      vst1q_f32(T[4][i], v_T4);
      vst1q_f32(T[5][i], v_W2);
  }
  for (int i = 0; i < 6; i++) {
      float32x4_t v_T0 = vld1q_f32(T[i][0]);
      float32x4_t v_T1 = vld1q_f32(T[i][1]);
      float32x4_t v_T2 = vld1q_f32(T[i][2]);

      float32x4_t v_t0 = vmulq_f32(v_01666, v_T2);
      float32x4_t v_t1 = vsubq_f32(vmulq_f32(v_minus_01666, v_T0), v_t0);
      float32x4_t v_t2 = vfmaq_f32(v_t0, v_004166, v_T0);

      float32x4_t v_WTM0 = vmulq_f32(v_025, v_T0);
      float32x4_t v_WTM1 = vfmaq_f32(v_t1, v_minus_01666, v_T1);
      float32x4_t v_WTM2 = vfmaq_f32(v_t1, v_01666, v_T1);
      float32x4_t v_WTM3 = vfmaq_f32(v_t2, v_00833, v_T1);
      float32x4_t v_WTM4 = vfmaq_f32(v_t2, v_minus_00833, v_T1);

      vst1q_f32(WTM[i*6+0], v_WTM0);
      vst1q_f32(WTM[i*6+1], v_WTM1);
      vst1q_f32(WTM[i*6+2], v_WTM2);
      vst1q_f32(WTM[i*6+3], v_WTM3);
      vst1q_f32(WTM[i*6+4], v_WTM4);
      vst1q_f32(WTM[i*6+5], v_T2);
  }
}


void trans_O_4x4_3x3(float* OTM[36], float* O[16], float* bias,
    int h, int w, int _pad_h_mod_4, int _pad_w_mod_4, int oh, int ow)
{
  float T[4][6][8];
  // bias
  float32x4_t v_b = vld1q_f32(bias);

  float32x4_t v_0 = vmovq_n_f32(0);
  float32x4_t v_2 = vmovq_n_f32(2);
  float32x4_t v_4 = vmovq_n_f32(4);
  float32x4_t v_8 = vmovq_n_f32(8);

  for (int i = 0; i < 6; i++) {
      float32x4_t v_OTM0 = vld1q_f32(OTM[i]);
      float32x4_t v_OTM1 = vld1q_f32(OTM[1*6+i]);
      float32x4_t v_OTM2 = vld1q_f32(OTM[2*6+i]);
      float32x4_t v_OTM3 = vld1q_f32(OTM[3*6+i]);
      float32x4_t v_OTM4 = vld1q_f32(OTM[4*6+i]);
      float32x4_t v_OTM5 = vld1q_f32(OTM[5*6+i]);

      float32x4_t v_t0 = vaddq_f32(v_OTM1, v_OTM2);
      float32x4_t v_t1 = vaddq_f32(v_OTM3, v_OTM4);
      float32x4_t v_t2 = vsubq_f32(v_OTM1, v_OTM2);
      float32x4_t v_t3 = vsubq_f32(v_OTM3, v_OTM4);

      float32x4_t v_T0 = vaddq_f32(vaddq_f32(v_t0, v_t1), v_OTM0);
      float32x4_t v_T1 = vfmaq_f32(v_t2, v_t3, v_2);
      float32x4_t v_T2 = vfmaq_f32(v_t0, v_t1, v_4);
      float32x4_t v_T3 = vaddq_f32(vfmaq_f32(v_t2, v_t3, v_8), v_OTM5);

      vst1q_f32(T[0][i], v_T0);
      vst1q_f32(T[1][i], v_T1);
      vst1q_f32(T[2][i], v_T2);
      vst1q_f32(T[3][i], v_T3);
  }

  int pad_h_mod_4 = 0, pad_w_mod_4 = 0;
  if (h == oh && w == ow) {
      pad_h_mod_4 = _pad_h_mod_4;
      pad_w_mod_4 = _pad_w_mod_4;
  } else if (h == oh) {
      pad_h_mod_4 = _pad_h_mod_4;
  } else if (w == ow) {
      pad_w_mod_4 = _pad_w_mod_4;
  }

  for (int i = 0; i < 4 - pad_h_mod_4; i++) {
      float32x4_t v_T0 = vld1q_f32(T[i][0]);
      float32x4_t v_T1 = vld1q_f32(T[i][1]);
      float32x4_t v_T2 = vld1q_f32(T[i][2]);
      float32x4_t v_T3 = vld1q_f32(T[i][3]);
      float32x4_t v_T4 = vld1q_f32(T[i][4]);
      float32x4_t v_T5 = vld1q_f32(T[i][5]);

      float32x4_t v_t0 = vaddq_f32(v_T1, v_T2);
      float32x4_t v_t1 = vaddq_f32(v_T3, v_T4);
      float32x4_t v_t2 = vsubq_f32(v_T1, v_T2);
      float32x4_t v_t3 = vsubq_f32(v_T3, v_T4);

      float32x4_t v_O0 = vaddq_f32(vaddq_f32(v_t0, v_t1), v_T0);
      float32x4_t v_O1 = vfmaq_f32(v_t2, v_t3, v_2);
      float32x4_t v_O2 = vfmaq_f32(v_t0, v_t1, v_4);
      float32x4_t v_O3 = vaddq_f32(vfmaq_f32(v_t2, v_t3, v_8), v_T5);

      if (pad_w_mod_4 == 0) {
          vst1q_f32(O[i*4+0], vaddq_f32(v_O0, v_b));
          vst1q_f32(O[i*4+1], vaddq_f32(v_O1, v_b));
          vst1q_f32(O[i*4+2], vaddq_f32(v_O2, v_b));
          vst1q_f32(O[i*4+3], vaddq_f32(v_O3, v_b));
      } else if (pad_w_mod_4 == 1) {
          vst1q_f32(O[i*4+0], vaddq_f32(v_O0, v_b));
          vst1q_f32(O[i*4+1], vaddq_f32(v_O1, v_b));
          vst1q_f32(O[i*4+2], vaddq_f32(v_O2, v_b));
      } else if (pad_w_mod_4 == 2) {
          vst1q_f32(O[i*4+0], vaddq_f32(v_O0, v_b));
          vst1q_f32(O[i*4+1], vaddq_f32(v_O1, v_b));
      } else if (pad_w_mod_4 == 3) {
          vst1q_f32(O[i*4+0], vaddq_f32(v_O0, v_b));
      }
  }
}


void trans_I_4x4_3x3(float* ITM[36], float* I[36])
{
  float T[6][6][8];

  float32x4_t v_4 = vmovq_n_f32(4);
  float32x4_t v_minus_4 = vmovq_n_f32(-4);
  float32x4_t v_2 = vmovq_n_f32(2);
  float32x4_t v_minus_5 = vmovq_n_f32(-5);

  for (int i = 0; i < 6; i++) {
      float32x4_t v_I0 = vld1q_f32(I[0*6+i]);
      float32x4_t v_I1 = vld1q_f32(I[1*6+i]);
      float32x4_t v_I2 = vld1q_f32(I[2*6+i]);
      float32x4_t v_I3 = vld1q_f32(I[3*6+i]);
      float32x4_t v_I4 = vld1q_f32(I[4*6+i]);
      float32x4_t v_I5 = vld1q_f32(I[5*6+i]);

      float32x4_t v_t0 = vfmaq_f32(v_I4, v_I2, v_minus_4);
      float32x4_t v_t1 = vfmaq_f32(v_I3, v_I1, v_minus_4);
      float32x4_t v_t2 = vsubq_f32(v_I4, v_I2);
      float32x4_t v_t3 = vmulq_f32(vsubq_f32(v_I3, v_I1), v_2);
      float32x4_t v_t4 = vfmaq_f32(v_I4, v_I0, v_4);
      float32x4_t v_t5 = vfmaq_f32(v_I5, v_I1, v_4);

      float32x4_t v_T0 = vfmaq_f32(v_t4, v_I2, v_minus_5);
      float32x4_t v_T1 = vaddq_f32(v_t1, v_t0);
      float32x4_t v_T2 = vsubq_f32(v_t0, v_t1);
      float32x4_t v_T3 = vaddq_f32(v_t3, v_t2);
      float32x4_t v_T4 = vsubq_f32(v_t2, v_t3);
      float32x4_t v_T5 = vfmaq_f32(v_t5, v_I3, v_minus_5);

      vst1q_f32(T[0][i], v_T0);
      vst1q_f32(T[1][i], v_T1);
      vst1q_f32(T[2][i], v_T2);
      vst1q_f32(T[3][i], v_T3);
      vst1q_f32(T[4][i], v_T4);
      vst1q_f32(T[5][i], v_T5);
  }

  for (int i = 0; i < 6; i++) {
      float32x4_t v_T0 = vld1q_f32(T[i][0]);
      float32x4_t v_T1 = vld1q_f32(T[i][1]);
      float32x4_t v_T2 = vld1q_f32(T[i][2]);
      float32x4_t v_T3 = vld1q_f32(T[i][3]);
      float32x4_t v_T4 = vld1q_f32(T[i][4]);
      float32x4_t v_T5 = vld1q_f32(T[i][5]);

      float32x4_t v_t0 = vfmaq_f32(v_T4, v_T2, v_minus_4);
      float32x4_t v_t1 = vfmaq_f32(v_T3, v_T1, v_minus_4);
      float32x4_t v_t2 = vsubq_f32(v_T4, v_T2);
      float32x4_t v_t3 = vmulq_f32(vsubq_f32(v_T3, v_T1), v_2);
      float32x4_t v_t4 = vfmaq_f32(v_T4, v_T0, v_4);
      float32x4_t v_t5 = vfmaq_f32(v_T5, v_T1, v_4);

      float32x4_t v_ITM0 = vfmaq_f32(v_t4, v_T2, v_minus_5);
      float32x4_t v_ITM1 = vaddq_f32(v_t1, v_t0);
      float32x4_t v_ITM2 = vsubq_f32(v_t0, v_t1);
      float32x4_t v_ITM3 = vaddq_f32(v_t3, v_t2);
      float32x4_t v_ITM4 = vsubq_f32(v_t2, v_t3);
      float32x4_t v_ITM5 = vfmaq_f32(v_t5, v_T3, v_minus_5);
      
      vst1q_f32(ITM[i*6+0], v_ITM0);
      vst1q_f32(ITM[i*6+1], v_ITM1);
      vst1q_f32(ITM[i*6+2], v_ITM2);
      vst1q_f32(ITM[i*6+3], v_ITM3);
      vst1q_f32(ITM[i*6+4], v_ITM4);
      vst1q_f32(ITM[i*6+5], v_ITM5);
  }
}


void weight_trans_wino(float *weight, float *weight_tm, int ic, int oc, int fh, int fw)
{
  if (fh != 3 || fw != 3) {
    std::cerr << "weight_trans_wino() fh fw not equal to 3.\n";
    return;
  }
  for (int o = 0; o < oc/8; o++) {
    for (int c = 0; c < ic; c++) {
      int wgt_off_0 = (o*8)*ic*3*3 + c*3*3;
      int wgt_off_1 = (o*8+4)*ic*3*3 + c*3*3;
      int wtm_off_0 = o*6*6*ic*8 + c*8;
      int wtm_off_1 = o*6*6*ic*8 + c*8 + 4;
      float W[9][4];
      float *W_ptr[9];
      float *WTM[36];
      for (int hw = 0; hw < 9; hw++) {
        for (int o4 = 0; o4 < 4; o4++) {
          W[hw][o4] = weight[wgt_off_0 + hw + o4*ic*3*3];
        }
        W_ptr[hw] = W[hw];
      }
      for (int hw = 0; hw < 36; hw++) {
        WTM[hw] = weight_tm + wtm_off_0 + hw*ic*8;
      }
      trans_W_4x4_3x3(WTM, W_ptr);
      for (int hw = 0; hw < 9; hw++) {
        for (int o4 = 0; o4 < 4; o4++) {
          W[hw][o4] = weight[wgt_off_1 + hw + o4*ic*3*3];
        }
        W_ptr[hw] = W[hw];
      }
      for (int hw = 0; hw < 36; hw++) {
        WTM[hw] = weight_tm + wtm_off_1 + hw*ic*8;
      }
      trans_W_4x4_3x3(WTM, W_ptr);
    }
  }
}


void conv3x3s1p1_wino_one_step(float *input, float *weight, float *output, float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf)
{
  float* in_pad = buf;
  int ohow = oh*ow;
  ic /= 4;
  oc /= 4;
  int tile_h = (oh + 3) / 4;
  int tile_w = (ow + 3) / 4;
  int tiles = tile_h * tile_w;  // num of 6x6 blocks
  int p_left = p;
  int p_w_mod_4 = tile_w*4 - ow;
  int p_right = p + p_w_mod_4;
  int p_top = p;
  int p_h_mod_4 = tile_h*4 - oh;
  int p_bottom = p + p_h_mod_4;
  int ih_pad = ih + p_top + p_bottom;   // pad to 4|ih_pad(iw_pad) for 4x4 block input transform to 6x6 block
  int iw_pad = iw + p_left + p_right;

  // itm: 6*6*C*c4*hw8
  // otm: O*6*6*hw8*o8
  float* itm = buf + ic*ih_pad*iw_pad*4;
  float* otm = itm + 6*6*ic*4*8;
  for (int n = 0; n < nb; n++) {
    // copy into buff as input_pad
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

    // handle 8|tiles
    for (int hw = 0; hw < tiles-7; hw+=8) {
      // input transform
      // NCHWc4 => (6*6)*C*c4*hw8
      {
        float *ITM_ptr[36];
        float ITM0[36][4];
        float ITM1[36][4];
        float ITM2[36][4];
        float ITM3[36][4];
        float ITM4[36][4];
        float ITM5[36][4];
        float ITM6[36][4];
        float ITM7[36][4];
        float *I0[36];
        float *I1[36];
        float *I2[36];
        float *I3[36];
        float *I4[36];
        float *I5[36];
        float *I6[36];
        float *I7[36];
        int h0 = (hw/tile_w)*4;
        int w0 = (hw%tile_w)*4;
        int h1 = ((hw+1)/tile_w)*4;
        int w1 = ((hw+1)%tile_w)*4;
        int h2 = ((hw+2)/tile_w)*4;
        int w2 = ((hw+2)%tile_w)*4;
        int h3 = ((hw+3)/tile_w)*4;
        int w3 = ((hw+3)%tile_w)*4;
        int h4 = ((hw+4)/tile_w)*4;
        int w4 = ((hw+4)%tile_w)*4;
        int h5 = ((hw+5)/tile_w)*4;
        int w5 = ((hw+5)%tile_w)*4;
        int h6 = ((hw+6)/tile_w)*4;
        int w6 = ((hw+6)%tile_w)*4;
        int h7 = ((hw+7)/tile_w)*4;
        int w7 = ((hw+7)%tile_w)*4;
        for (int c = 0; c < ic; c++) {
          float* in_mov = in_pad + c*ih_pad*iw_pad*4;
          for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
              I0[i*6+j] = in_mov + (h0+i)*iw_pad*4 + (w0+j)*4;
              I1[i*6+j] = in_mov + (h1+i)*iw_pad*4 + (w1+j)*4;
              I2[i*6+j] = in_mov + (h2+i)*iw_pad*4 + (w2+j)*4;
              I3[i*6+j] = in_mov + (h3+i)*iw_pad*4 + (w3+j)*4;
              I4[i*6+j] = in_mov + (h4+i)*iw_pad*4 + (w4+j)*4;
              I5[i*6+j] = in_mov + (h5+i)*iw_pad*4 + (w5+j)*4;
              I6[i*6+j] = in_mov + (h6+i)*iw_pad*4 + (w6+j)*4;
              I7[i*6+j] = in_mov + (h7+i)*iw_pad*4 + (w7+j)*4;
            }
          }

          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM0[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I0);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM1[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I1);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM2[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I2);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM3[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I3);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM4[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I4);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM5[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I5);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM6[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I6);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM7[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I7);

          float* itm_mov = itm + c*4*8;

          for (int i = 0; i < 36; i++) {
            float* itm_c4hw8 = itm_mov + i*ic*4*8;
            float32x4_t v0 = vld1q_f32(ITM0[i]);
            float32x4_t v1 = vld1q_f32(ITM1[i]);
            float32x4_t v2 = vld1q_f32(ITM2[i]);
            float32x4_t v3 = vld1q_f32(ITM3[i]);
            float32x4_t v4 = vld1q_f32(ITM4[i]);
            float32x4_t v5 = vld1q_f32(ITM5[i]);
            float32x4_t v6 = vld1q_f32(ITM6[i]);
            float32x4_t v7 = vld1q_f32(ITM7[i]);
            vst1q_f32(itm_c4hw8,
                vzip1q_f32(
                    vzip1q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                    vzip1q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
            vst1q_f32(itm_c4hw8 + 4,
                vzip2q_f32(
                    vzip1q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                    vzip1q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
            vst1q_f32(itm_c4hw8 + 4*2,
                vzip1q_f32(
                    vzip2q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                    vzip2q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
            vst1q_f32(itm_c4hw8 + 4*3,
                vzip2q_f32(
                    vzip2q_f32(vzip1q_f32(v0, v4), vzip1q_f32(v2, v6)),
                    vzip2q_f32(vzip1q_f32(v1, v5), vzip1q_f32(v3, v7))));
            vst1q_f32(itm_c4hw8 + 4*4,
                vzip1q_f32(
                    vzip1q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                    vzip1q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
            vst1q_f32(itm_c4hw8 + 4*5,
                vzip2q_f32(
                    vzip1q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                    vzip1q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
            vst1q_f32(itm_c4hw8 + 4*6,
                vzip1q_f32(
                    vzip2q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                    vzip2q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
            vst1q_f32(itm_c4hw8 + 4*7,
                vzip2q_f32(
                    vzip2q_f32(vzip2q_f32(v0, v4), vzip2q_f32(v2, v6)),
                    vzip2q_f32(vzip2q_f32(v1, v5), vzip2q_f32(v3, v7))));
          }
        }
      }

      for (int o = 0; o < oc; o+=2) {
        float* b_0 = bias + o*4;
        float* b_1 = b_0 + 4;
        float zero[4] = {0, 0, 0, 0};
        float* wgt_mov = weight + o*6*6*ic*4*4;   // note that o*o4 here, not o*o8
        float* otm_mov = otm + o*6*6*8*4;   // note that o*o4 here, not o*o8

        // dot prod
        // (6*6)*C*c4*hw8 times O*(6*6)*C*c4*o8 = O*(6*6)*hw8*o8
        // in fact output layout is O*(6*6)*O2*hw8*o4 (by sgemm)
        // or can implement a similar sgemm in hw8*o8
        for (int i = 0; i < 36; i++) {
          float* itm_c4hw8 = itm + i*ic*4*8;
          float* wgt_c4o8 = wgt_mov + i*ic*4*8;
          float* otm_hw8o8 = otm_mov + i*8*8;
          sgemm_8x8(itm_c4hw8, wgt_c4o8, otm_hw8o8, otm_hw8o8 + 8*4, zero, zero, ic*4);
        }

        // output transform
        // O*(6*6)*O2*hw8*o4 => NOHWo4
        for (int hw8 = 0; hw8 < 8; hw8++) {
          int h = (hw+hw8) / tile_w;
          int w = (hw+hw8) % tile_w;
          float* out_0 = output + n*oc*oh*ow*4 + o*oh*ow*4 + h*4*ow*4 + w*4*4;
          int otm_off_0 = o*36*8*4 + hw8*4;   // note that o*o4 here, not o*o8

          float *OTM0[36];
          float *O0[16];
          for (int idx = 0; idx < 36; idx++) {
            OTM0[idx] = otm + otm_off_0 + idx*2*8*4;
          }
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              O0[i*4 + j] = out_0 + i*ow*4 + j*4;
            }
          }
          trans_O_4x4_3x3(OTM0, O0, b_0, h, w, p_h_mod_4, p_w_mod_4, tile_h-1, tile_w-1);
        }
        
        for (int hw8 = 0; hw8 < 8; hw8++) {
          int h = (hw+hw8) / tile_w;
          int w = (hw+hw8) % tile_w;
          float* out_1 = output + n*oc*oh*ow*4 + (o+1)*oh*ow*4 + h*4*ow*4 + w*4*4;
          int otm_off_1 = o*36*8*4 + 8*4 + hw8*4;   // note that o*o4 here, not o*o8

          float *OTM1[36];
          float *O1[16];
          for (int idx = 0; idx < 36; idx++) {
            OTM1[idx] = otm + otm_off_1 + idx*2*8*4;
          }
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              O1[i*4 + j] = out_1 + i*ow*4 + j*4;
            }
          }
          trans_O_4x4_3x3(OTM1, O1, b_1, h, w, p_h_mod_4, p_w_mod_4, tile_h-1, tile_w-1);
        }
      }
    }

    // handle 4|(tiles%8)
    int tiles_start = (tiles/8)*8;
    for (int hw = tiles_start; hw < tiles-3; hw+=4) {
      // input transform
      // NCHWc4 => (6*6)*C*c4*hw4
      {
        float *ITM_ptr[36];
        float ITM0[36][4];
        float ITM1[36][4];
        float ITM2[36][4];
        float ITM3[36][4];
        float *I0[36];
        float *I1[36];
        float *I2[36];
        float *I3[36];
        int h0 = (hw/tile_w)*4;
        int w0 = (hw%tile_w)*4;
        int h1 = ((hw+1)/tile_w)*4;
        int w1 = ((hw+1)%tile_w)*4;
        int h2 = ((hw+2)/tile_w)*4;
        int w2 = ((hw+2)%tile_w)*4;
        int h3 = ((hw+3)/tile_w)*4;
        int w3 = ((hw+3)%tile_w)*4;
        for (int c = 0; c < ic; c++) {
          float* in_mov = in_pad + c*ih_pad*iw_pad*4;
          for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
              I0[i*6+j] = in_mov + (h0+i)*iw_pad*4 + (w0+j)*4;
              I1[i*6+j] = in_mov + (h1+i)*iw_pad*4 + (w1+j)*4;
              I2[i*6+j] = in_mov + (h2+i)*iw_pad*4 + (w2+j)*4;
              I3[i*6+j] = in_mov + (h3+i)*iw_pad*4 + (w3+j)*4;
            }
          }

          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM0[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I0);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM1[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I1);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM2[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I2);
          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM3[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I3);

          float* itm_mov = itm + c*4*4;

          for (int i = 0; i < 36; i++) {
            float* itm_c4hw8 = itm_mov + i*ic*4*4;
            __asm__ __volatile__(
              "ldr q0, [%[in_0]]\n"
              "ldr q1, [%[in_1]]\n"
              "ldr q2, [%[in_2]]\n"
              "ldr q3, [%[in_3]]\n"
              "st4 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[itm]]\n"
              :[itm]"+r"(itm_c4hw8)
              :[in_0]"r"(ITM0[i]),
               [in_1]"r"(ITM1[i]),
               [in_2]"r"(ITM2[i]),
               [in_3]"r"(ITM3[i])
              :"memory", "cc", "q0", "q1", "q2", "q3"
            );
          }
        }
      }

      for (int o = 0; o < oc; o+=2) {
        float* b_0 = bias + o*4;
        float* b_1 = b_0 + 4;
        float zero[4] = {0, 0, 0, 0};
        float* wgt_mov = weight + o*6*6*ic*4*4;   // note that o*o4 here, not o*o8
        float* otm_mov = otm + o*6*6*4*4;   // note that o*o4 here, not o*o8

        // dot prod
        // (6*6)*C*c4*hw4 times O*(6*6)*C*c4*o8 = O*(6*6)*hw4*o8
        // in fact output layout is O*(6*6)*O2*hw4*o4 (by sgemm)
        // or can implement a similar sgemm in hw4*o8
        for (int i = 0; i < 36; i++) {
          float* itm_c4hw4 = itm + i*ic*4*4;
          float* wgt_c4o8 = wgt_mov + i*ic*4*8;
          float* otm_hw4o8 = otm_mov + i*4*8;
          sgemm_4x8(itm_c4hw4, wgt_c4o8, otm_hw4o8, otm_hw4o8 + 4*4, zero, zero, ic*4);
        }

        // output transform
        // O*(6*6)*O2*hw4*o4 => NOHWo4
        for (int hw4 = 0; hw4 < 4; hw4++) {
          int h = (hw+hw4) / tile_w;
          int w = (hw+hw4) % tile_w;
          float* out_0 = output + n*oc*oh*ow*4 + o*oh*ow*4 + h*4*ow*4 + w*4*4;
          int otm_off_0 = o*36*4*4 + hw4*4;   // note that o*o4 here, not o*o8

          float *OTM0[36];
          float *O0[16];
          for (int idx = 0; idx < 36; idx++) {
            OTM0[idx] = otm + otm_off_0 + idx*2*4*4;
          }
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              O0[i*4 + j] = out_0 + i*ow*4 + j*4;
            }
          }
          trans_O_4x4_3x3(OTM0, O0, b_0, h, w, p_h_mod_4, p_w_mod_4, tile_h-1, tile_w-1);
        }
        
        for (int hw4 = 0; hw4 < 4; hw4++) {
          int h = (hw+hw4) / tile_w;
          int w = (hw+hw4) % tile_w;
          float* out_1 = output + n*oc*oh*ow*4 + (o+1)*oh*ow*4 + h*4*ow*4 + w*4*4;
          int otm_off_1 = o*36*4*4 + 4*4 + hw4*4;   // note that o*o4 here, not o*o8

          float *OTM1[36];
          float *O1[16];
          for (int idx = 0; idx < 36; idx++) {
            OTM1[idx] = otm + otm_off_1 + idx*2*4*4;
          }
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              O1[i*4 + j] = out_1 + i*ow*4 + j*4;
            }
          }
          trans_O_4x4_3x3(OTM1, O1, b_1, h, w, p_h_mod_4, p_w_mod_4, tile_h-1, tile_w-1);
        }
      }
    }

    // handle tiles%4
    tiles_start = (tiles/4)*4;
    for (int hw = tiles_start; hw < tiles; hw++) {
      // input transform
      // NCHWc4 => (6*6)*C*c4*hw1
      {
        float *ITM_ptr[36];
        float ITM0[36][4];
        float *I0[36];
        int h0 = (hw/tile_w)*4;
        int w0 = (hw%tile_w)*4;
        for (int c = 0; c < ic; c++) {
          float* in_mov = in_pad + c*ih_pad*iw_pad*4;
          for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
              I0[i*6+j] = in_mov + (h0+i)*iw_pad*4 + (w0+j)*4;
            }
          }

          for (int i = 0; i < 36; i++) {
              ITM_ptr[i] = ITM0[i];
          }
          trans_I_4x4_3x3(ITM_ptr, I0);

          float* itm_mov = itm + c*4*1;

          for (int i = 0; i < 36; i++) {
            float* itm_c4hw1 = itm_mov + i*ic*4*1;
            memcpy(itm_c4hw1, ITM0[i], 4*sizeof(float));
          }
        }
      }

      for (int o = 0; o < oc; o+=2) {
        float* b_0 = bias + o*4;
        float* b_1 = b_0 + 4;
        float zero[4] = {0, 0, 0, 0};
        float* wgt_mov = weight + o*6*6*ic*4*4;   // note that o*o4 here, not o*o8
        float* otm_mov = otm + o*6*6*1*4;   // note that o*o4 here, not o*o8

        // dot prod
        // (6*6)*C*c4*hw1 times O*(6*6)*C*c4*o8 = O*(6*6)*hw1*o8
        // in fact output layout is O*(6*6)*O2*hw1*o4 (by sgemm)
        // or can implement a similar sgemm in hw1*o8
        for (int i = 0; i < 36; i++) {
          float* itm_c4hw1 = itm + i*ic*4*1;
          float* wgt_c4o8 = wgt_mov + i*ic*4*8;
          float* otm_hw1o8 = otm_mov + i*1*8;
          sgemm_1x8(itm_c4hw1, wgt_c4o8, otm_hw1o8, otm_hw1o8 + 1*4, zero, zero, ic*4);
        }

        // output transform
        // O*(6*6)*O2*hw1*o4 => NOHWo4
        for (int hw1 = 0; hw1 < 1; hw1++) {
          int h = (hw+hw1) / tile_w;
          int w = (hw+hw1) % tile_w;
          float* out_0 = output + n*oc*oh*ow*4 + o*oh*ow*4 + h*4*ow*4 + w*4*4;
          int otm_off_0 = o*36*1*4 + hw1*4;   // note that o*o4 here, not o*o8

          float *OTM0[36];
          float *O0[16];
          for (int idx = 0; idx < 36; idx++) {
            OTM0[idx] = otm + otm_off_0 + idx*2*1*4;
          }
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              O0[i*4 + j] = out_0 + i*ow*4 + j*4;
            }
          }
          trans_O_4x4_3x3(OTM0, O0, b_0, h, w, p_h_mod_4, p_w_mod_4, tile_h-1, tile_w-1);
        }
        
        for (int hw1 = 0; hw1 < 1; hw1++) {
          int h = (hw+hw1) / tile_w;
          int w = (hw+hw1) % tile_w;
          float* out_1 = output + n*oc*oh*ow*4 + (o+1)*oh*ow*4 + h*4*ow*4 + w*4*4;
          int otm_off_1 = o*36*1*4 + 1*4 + hw1*4;   // note that o*o4 here, not o*o8

          float *OTM1[36];
          float *O1[16];
          for (int idx = 0; idx < 36; idx++) {
            OTM1[idx] = otm + otm_off_1 + idx*2*1*4;
          }
          for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
              O1[i*4 + j] = out_1 + i*ow*4 + j*4;
            }
          }
          trans_O_4x4_3x3(OTM1, O1, b_1, h, w, p_h_mod_4, p_w_mod_4, tile_h-1, tile_w-1);
        }
      }
    }
  }
}


#endif // USE_NEON_A55 || USE_NEON_A76