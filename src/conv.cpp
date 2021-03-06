#include "conv.hpp"


// More description see header for detail.


void get_output_hw(int ih, int iw, int fh, int fw, int s, int p, int *oh, int *ow)
{
  // ignore remainder if it appears divided by s, like what Caffe does.
  *oh = (ih + 2*p - fh) / s + 1;
  *ow = (iw + 2*p - fw) / s + 1;
}


void infer_conv_alg(int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, ConvAlg *alg)
{
  // TODO make implementations more general when there are remainders.
  if (ic%4 != 0 || oc%8 != 0) {
    *alg = CONV_NOT_MATCH;
    return;
  }

#if defined(_USE_NEON_A55) || defined(_USE_NEON_A76)
  if (fh==3 && fw==3 && s==1 && p==1) {
    *alg = CONV_WINO_ONE_STEP;
  } else if (ih < 24 && iw < 24) {
#else
  if (ih < 24 && iw < 24) {     // just guess threshold (between 14 and 28)
#endif
    *alg = CONV_IM2COL_TOTAL_PACK;
  } else {
    *alg = CONV_IM2COL_TILE_PACK;
  }
}


void conv_buffer_size(int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, ConvAlg alg, int *bytes)
{
  switch(alg) {
    case CONV_WINO_ONE_STEP: {
      int tile_h = (oh + 3) / 4;
      int tile_w = (ow + 3) / 4;
      int p_left = p;
      int p_w_mod_4 = tile_w*4 - ow;
      int p_right = p + p_w_mod_4;
      int p_top = p;
      int p_h_mod_4 = tile_h*4 - oh;
      int p_bottom = p + p_h_mod_4;
      int ih_pad = ih + p_top + p_bottom;   // pad to 4|ih_pad(iw_pad) for 4x4 block input transform to 6x6 block
      int iw_pad = iw + p_left + p_right;
      *bytes = ic*ih_pad*iw_pad + 6*6*ic*8 + oc*6*6*8;
      break;
    }
    case CONV_IM2COL_TOTAL_PACK: {
      // padding + in_pack
      int ih_pad = ih + 2*p;
      int iw_pad = iw + 2*p;
      *bytes = ic*ih_pad*iw_pad + fh*fw*ic*oh*ow;
      break;
    }
    case CONV_IM2COL_TILE_PACK: {
      // padding + in_pack
      int ih_pad = ih + 2*p;
      int iw_pad = iw + 2*p;
      *bytes = ic*ih_pad*iw_pad + fh*fw*ic*8;
      break;
    }
    default: {
      break;
    }
  }
  *bytes *= sizeof(float);
}


void weight_trans_size(int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, ConvAlg alg, int *bytes)
{
  switch(alg) {
    case CONV_WINO_ONE_STEP: {
      *bytes = ic*oc*6*6;
      break;
    }
    default: {
      *bytes = ic*oc*fh*fw;
      break;
    }
  }
  *bytes *= sizeof(float);
}


void weight_trans(float *weight, float *weight_tm, int ic, int oc, int fh, int fw, ConvAlg alg)
{
  switch(alg) {
    case CONV_IM2COL_TOTAL_PACK: {
      weight_trans_im2col(weight, weight_tm, ic, oc, fh, fw);
      break;
    }
    case CONV_IM2COL_TILE_PACK: {
      weight_trans_im2col(weight, weight_tm, ic, oc, fh, fw);
      break;
    }
    case CONV_WINO_ONE_STEP: {
      weight_trans_wino(weight, weight_tm, ic, oc, fh, fw);
      break;
    }
    default: {
      break;
    }
  }
}


void conv(float *input, float *weight, float *output, float *bias,
          int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf, ConvAlg alg)
{
  switch(alg) {
    case CONV_WINO_ONE_STEP: {
      conv3x3s1p1_wino_one_step(input, weight, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf);
      break;
    }
    case CONV_IM2COL_TOTAL_PACK: {
      conv_im2col_total_pack(input, weight, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf);
      break;
    }
    case CONV_IM2COL_TILE_PACK: {
      conv_im2col_tile_pack(input, weight, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf);
      break;
    }
    default: {
      break;
    }
  }
}