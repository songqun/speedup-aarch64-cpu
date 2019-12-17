#include "conv.hpp"
#include "sgemm.hpp"
#include <stdlib.h>
#include <string.h>

#if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
#include <arm_neon.h>
#endif


// More description see header for detail.


inline void trans_W_4x4_3x3(float* Fw[36], float* F[9])
{
  float T[6][3][8];

  float32x4_t v_01666 = vmovq_n_f32(0.1666666666666667f);
  float32x4_t v_minus_01666 = vmovq_n_f32(-0.1666666666666667f);
  float32x4_t v_00833 = vmovq_n_f32(0.0833333333333333f);
  float32x4_t v_minus_00833 = vmovq_n_f32(-0.0833333333333333f);
  float32x4_t v_004166 = vmovq_n_f32(0.0416666666666667f);
  float32x4_t v_025 = vmovq_n_f32(0.25f);

  for (int i = 0; i < 3; i++) {
      float32x4_t v_F0 = vld1q_f32(F[0*3+i]);
      float32x4_t v_F1 = vld1q_f32(F[1*3+i]);
      float32x4_t v_F2 = vld1q_f32(F[2*3+i]);

      float32x4_t v_t0 = vmulq_f32(v_01666, v_F2);
      float32x4_t v_t1 = vsubq_f32(vmulq_f32(v_minus_01666, v_F0), v_t0);
      float32x4_t v_t2 = vfmaq_f32(v_t0, v_004166, v_F0);

      float32x4_t v_T0 = vmulq_f32(v_025, v_F0);
      float32x4_t v_T1 = vfmaq_f32(v_t1, v_minus_01666, v_F1);
      float32x4_t v_T2 = vfmaq_f32(v_t1, v_01666, v_F1);
      float32x4_t v_T3 = vfmaq_f32(v_t2, v_00833, v_F1);
      float32x4_t v_T4 = vfmaq_f32(v_t2, v_minus_00833, v_F1);

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

      float32x4_t v_Fw0 = vmulq_f32(v_025, v_T0);
      float32x4_t v_Fw1 = vfmaq_f32(v_t1, v_minus_01666, v_T1);
      float32x4_t v_Fw2 = vfmaq_f32(v_t1, v_01666, v_T1);
      float32x4_t v_Fw3 = vfmaq_f32(v_t2, v_00833, v_T1);
      float32x4_t v_Fw4 = vfmaq_f32(v_t2, v_minus_00833, v_T1);

      vst1q_f32(Fw[i*6+0], v_Fw0);
      vst1q_f32(Fw[i*6+1], v_Fw1);
      vst1q_f32(Fw[i*6+2], v_Fw2);
      vst1q_f32(Fw[i*6+3], v_Fw3);
      vst1q_f32(Fw[i*6+4], v_Fw4);
      vst1q_f32(Fw[i*6+5], v_T2);
  }
}


inline void trans_O_4x4_3x3(float* Ow[36], float* O[16], float* bias,
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
      float32x4_t v_Ow0 = vld1q_f32(Ow[i]);
      float32x4_t v_Ow1 = vld1q_f32(Ow[1*6+i]);
      float32x4_t v_Ow2 = vld1q_f32(Ow[2*6+i]);
      float32x4_t v_Ow3 = vld1q_f32(Ow[3*6+i]);
      float32x4_t v_Ow4 = vld1q_f32(Ow[4*6+i]);
      float32x4_t v_Ow5 = vld1q_f32(Ow[5*6+i]);

      float32x4_t v_t0 = vaddq_f32(v_Ow1, v_Ow2);
      float32x4_t v_t1 = vaddq_f32(v_Ow3, v_Ow4);
      float32x4_t v_t2 = vsubq_f32(v_Ow1, v_Ow2);
      float32x4_t v_t3 = vsubq_f32(v_Ow3, v_Ow4);

      float32x4_t v_T0 = vaddq_f32(vaddq_f32(v_t0, v_t1), v_Ow0);
      float32x4_t v_T1 = vfmaq_f32(v_t2, v_t3, v_2);
      float32x4_t v_T2 = vfmaq_f32(v_t0, v_t1, v_4);
      float32x4_t v_T3 = vaddq_f32(vfmaq_f32(v_t2, v_t3, v_8), v_Ow5);

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


inline void trans_I_4x4_3x3(float* Iw[36], float* I[36])
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

      float32x4_t v_Iw0 = vfmaq_f32(v_t4, v_T2, v_minus_5);
      float32x4_t v_Iw1 = vaddq_f32(v_t1, v_t0);
      float32x4_t v_Iw2 = vsubq_f32(v_t0, v_t1);
      float32x4_t v_Iw3 = vaddq_f32(v_t3, v_t2);
      float32x4_t v_Iw4 = vsubq_f32(v_t2, v_t3);
      float32x4_t v_Iw5 = vfmaq_f32(v_t5, v_T3, v_minus_5);
      
      vst1q_f32(Iw[i*6+0], v_Iw0);
      vst1q_f32(Iw[i*6+1], v_Iw1);
      vst1q_f32(Iw[i*6+2], v_Iw2);
      vst1q_f32(Iw[i*6+3], v_Iw3);
      vst1q_f32(Iw[i*6+4], v_Iw4);
      vst1q_f32(Iw[i*6+5], v_Iw5);
  }
}


void weight_trans_wino(float *weight, float *weight_tm, int ic, int oc, int fh, int fw)
{
  
}


void conv3x3s1p1_wino_two_step(float *input, float *weight, float *output, float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf)
{

}