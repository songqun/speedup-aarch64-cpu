#include "conv.hpp"
#include <iostream>
#include <string.h>


// From Caffe
void conv_naive(float *input, float *weight, float *output, float *bias,
                int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p)
{
  for (int n = 0; n < nb; n++) {
    for (int o = 0; o < oc; o++) {
      for (int c = 0; c < ic; c++) {
        for (int h = 0; h < oh; h++) {
          for (int w = 0; w < ow; w++) {
            for (int f_h = 0; f_h < fh; f_h++) {
              for (int f_w = 0; f_w < fw; f_w++) {
                int in_h = h * s - p + f_h;
                int in_w = w * s - p + f_w;
                if (in_h >= 0 && in_h < ih && in_w >= 0 && in_w < iw) {
                  output[nb*oc*oh*ow + o*oh*ow + h*ow + w] += 
                      input[nb*ic*ih*iw + c*ih*iw + in_h*iw + in_w] * 
                      weight[o*ic*fh*fw + c*fh*fw + f_h*fw + f_w];
                }
              }
            }
          }
        }
      }
      for (int hw = 0; hw < oh*ow; hw++) {
        output[nb*oc*oh*ow + o*oh*ow + hw] += bias[o];
      }
    }
  }
}


int main()
{
  // setup params
  int nb = 1, ic = 16, oc = 16, ih = 16, iw = 16, fh = 3, fw = 3, s = 1, p = 1;

  // setup input, weight, bias
  float *input = (float*)malloc(nb*ic*ih*iw*sizeof(float));
  float *input_ref = (float*)malloc(nb*ic*ih*iw*sizeof(float));
  for (int i = 0; i < nb*ic*ih*iw; i++) {
    input[i] = rand() % 5;
    input_ref[i] = input[i];
  }
  float *weight = (float*)malloc(oc*ic*fh*fw*sizeof(float));
  float *weight_ref = (float*)malloc(oc*ic*fh*fw*sizeof(float));
  for (int i = 0; i < oc*ic*fh*fw; i++) {
    weight[i] = rand() % 5;
    weight_ref[i] = weight[i];
  }
  float *bias = (float*)malloc(oc*sizeof(float));
  float *bias_ref = (float*)malloc(oc*sizeof(float));
  for (int i = 0; i < oc; i++) {
    bias[i] = rand() % 5;
    bias_ref[i] = bias[i];
  }

  // setup output
  int oh, ow;
  get_output_hw(ih, iw, fh, fw, s, p, &oh, &ow);
  float *output = (float*)malloc(nb*oc*oh*ow*sizeof(float));
  float *output_ref = (float*)malloc(nb*oc*oh*ow*sizeof(float));
  memset(output, 0, nb*oc*oh*ow*sizeof(float));
  memset(output_ref, 0, nb*oc*oh*ow*sizeof(float));

  // do conv_naive as reference
  conv_naive(input_ref, weight_ref, output_ref, bias_ref,
             nb, ic, ih, iw, oc, oh, ow ,fh, fw, s, p);



  free(input);
  free(input_ref);
  free(weight);
  free(weight_ref);
  free(output);
  free(output_ref);
}