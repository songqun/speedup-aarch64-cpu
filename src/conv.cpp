#include "conv.hpp"


// More description see header for detail.


void conv3x3s1p1_wino_two_step(const float *input, const float *weight, float *output, const float *bias,
                 int nb, int ic, int ih, int iw, int oh, int ow, int oc, int fh, int fw, int s, int p, float *buf)
{

}


void conv1x1s1p0(const float *input, const float *weight, float *output, const float *bias,
                 int nb, int ic, int ih, int iw, int oh, int ow, int oc, int fh, int fw, int s, int p, float *buf)
{

}


void conv_im2col(const float *input, const float *weight, float *output, const float *bias,
                 int nb, int ic, int ih, int iw, int oh, int ow, int oc, int fh, int fw, int s, int p, float *buf)
{

}


void get_output_hw(int ih, int iw, int fh, int fw, int s, int p, int *oh, int *ow)
{
  // ignore remainder if it appears divided by s, like what Caffe does.
  *oh = (ih + 2*p - fh) / s + 1;
  *ow = (iw + 2*p - fw) / s + 1;
}


void infer_conv_alg(int nb, int ic, int ih, int iw, int oh, int ow, int oc, int fh, int fw, int s, int p, ConvAlg *alg)
{
  if (fh!=fw || (oc%2)!=0) {
    *alg = CONV_NOT_MATCH;
  }

  // TODO make implementations more general when there are remainders.
  if (fh==1 && fw==1 && s==1 && p==0
      && (oh*ow)%8==0) {
    *alg = CONV_1X1S1P0;
  } else if (fh==3 && fw==3 && s==1 && p==1
             && oh%4==0 && ow%4==0 && (oh*ow)%128==0) {
    *alg = CONV_WINO_TWO_STEP;
  } else if (oh*ow%8==0){
    *alg = CONV_IM2COL;
  } else {
    *alg = CONV_NOT_MATCH;
  }
}


void conv_buffer_size(int nb, int ic, int ih, int iw, int oh, int ow, int oc, int fh, int fw, int s, int p, ConvAlg alg, int *bytes)
{
  switch(alg) {
    case CONV_1X1S1P0:
      // in_pack
      *bytes = (ic*4)*8;
      break;
    case CONV_WINO_TWO_STEP:
      *bytes = 0; //TODO
      break;
    case CONV_IM2COL: {
      // padding + in_pack
      int ih_pad = ih + 2*p;
      int iw_pad = iw + 2*p;
      *bytes = ic*ih_pad*iw_pad*4 + fh*fw*(ic*4)*8;
      break;
    }
    default:
      break;
  }
  *bytes *= sizeof(float);
}


void conv(const float *input, const float *weight, float *output, const float *bias,
          int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p, float *buf, ConvAlg alg)
{
  switch(alg) {
    case CONV_1X1S1P0:
      conv1x1s1p0(input, weight, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf);
      break;
    case CONV_WINO_TWO_STEP:
      conv3x3s1p1_wino_two_step(input, weight, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf);
      break;
    case CONV_IM2COL:
      conv_im2col(input, weight, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf);
      break;
    default:
      break;
  }
}