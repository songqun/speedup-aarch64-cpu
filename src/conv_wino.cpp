#include "conv.hpp"
#include "sgemm.hpp"
#include <stdlib.h>
#include <string.h>


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
      vst1q_f32(T[5][i], v_F2);
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

  for (U32 i = 0; i < 6; i++) {
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

  for (U32 i = 0; i < 6; i++) {
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
      int wtm_off_0 = o*36*ic*8 + c*8;
      int wtm_off_1 = o*36*ic*8 + c*8 + 4;
      float W[9][4];
      float *W_ptr[9];
      float *WTM[36];
      for (int hw = 0; hw < 9; hw++) {
        for (int o4 = 0; o4 < 4; o++) {
          W[hw][o4] = weight[wgt_off_0 + hw + o4*ic*fh*fw];
        }
        W_ptr[hw] = W[hw];
      }
      for (int hw = 0; hw < 36; hw++) {
        WTM[hw] = weight_tm + wtm_off_0 + hw*ic*8;
      }
      trans_W_4x4_3x3(WTM, W_ptr);
      for (int hw = 0; hw < 9; hw++) {
        for (int o4 = 0; o4 < 4; o++) {
          W[hw][o4] = weight[wgt_off_1 + hw + o4*ic*fh*fw];
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

}


#endif // USE_NEON_A55 || USE_NEON_A76