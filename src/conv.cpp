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
  if (fh==3 && fw==3 && s==1 && p==1
             && oh%4==0 && ow%4==0 && (oh*ow)%128==0) {
    *alg = CONV_WINO_TWO_STEP;
  } else {
    *alg = CONV_IM2COL_TOTAL_PACK; // OR TILE_PACK
  }
}


void conv_buffer_size(int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, ConvAlg alg, int *bytes)
{
  switch(alg) {
    case CONV_WINO_TWO_STEP: {
      *bytes = 0; //TODO
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
    case CONV_WINO_TWO_STEP: {
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
    case CONV_WINO_TWO_STEP: {
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
    case CONV_WINO_TWO_STEP: {
      conv3x3s1p1_wino_two_step(input, weight, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf);
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