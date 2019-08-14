#include "conv.hpp"
#include "util.hpp"
#include <iostream>
#include <string.h>
#include <math.h>


// From Caffe
void conv_naive(float *input, float *weight, float *output, float *bias,
                int nb, int ic, int ih, int iw, int oc, int oh, int ow, int fh, int fw, int s, int p)
{
  float *in = (float*)malloc(nb*ic*ih*iw*sizeof(float));
  float *out = (float*)malloc(nb*oc*oh*ow*sizeof(float));
  // NCHWc4 => NCHW
  for (int n = 0; n < nb; n++) {
    for (int c = 0; c < ic/4; c++) {
      for (int hw = 0; hw < ih*iw; hw++) {
        for (int c4 = 0; c4 < 4; c4++) {
          in[n*ic*ih*iw + (c*4+c4)*ih*iw + hw] = input[n*ic*ih*iw + c*ih*iw*4 + hw*4 + c4];
        }
      }
    }
  }
  // compute
  for (int n = 0; n < nb; n++) {
    for (int o = 0; o < oc; o++) {
      for (int hw = 0; hw < oh*ow; hw++) {
        out[n*oc*oh*ow + o*oh*ow + hw] = bias[n*oc + o];
      }
      for (int c = 0; c < ic; c++) {
        for (int h = 0; h < oh; h++) {
          for (int w = 0; w < ow; w++) {
            for (int f_h = 0; f_h < fh; f_h++) {
              for (int f_w = 0; f_w < fw; f_w++) {
                int in_h = h * s - p + f_h;
                int in_w = w * s - p + f_w;
                if (in_h >= 0 && in_h < ih && in_w >= 0 && in_w < iw) {
                  out[n*oc*oh*ow + o*oh*ow + h*ow + w] += 
                      in[n*ic*ih*iw + c*ih*iw + in_h*iw + in_w] * 
                      weight[o*ic*fh*fw + c*fh*fw + f_h*fw + f_w];
                }
              }
            }
          }
        }
      }
    }
  }
  // NOHW => NOHWo4
  for (int n = 0; n < nb; n++) {
    for (int o = 0; o < oc/4; o++) {
      for (int hw = 0; hw < oh*ow; hw++) {
        for (int o4 = 0; o4 < 4; o4++) {
          output[n*oc*oh*ow + o*oh*ow*4 + hw*4 + o4] = out[n*oc*oh*ow + (o*4+o4)*oh*ow + hw];
        }
      }
    }
  }
  free(in);
  free(out);
}


void compare(float *output, float *output_ref, float out_size)
{
  for (int i = 0; i < out_size; i++) {
    float err = abs(output[i] - output_ref[i]) / output_ref[i];
    if (err > 0.01) {
      std::cout << i << " : " << output[i] << " : " << output_ref[i] << " : " << (err*100) << "\n";
    }
  }
}


int main()
{
  // setup params
  int nb = 1, ic = 16, oc = 24, ih = 16, iw = 16, fh = 1, fw = 1, s = 2, p = 0;

  if (ic%4 != 0 || oc%8 != 0 || fh != fw) {
    std::cerr << "Not support.\n";
    return -1;
  }

  // setup input, weight, bias
  float *input = (float*)malloc(nb*ic*ih*iw*sizeof(float));
  float *input_ref = (float*)malloc(nb*ic*ih*iw*sizeof(float));
  for (int i = 0; i < nb*ic*ih*iw; i++) {
    //input[i] = rand() % 5;
    input[i] = 1;
    input_ref[i] = input[i];
  }
  float *weight = (float*)malloc(oc*ic*fh*fw*sizeof(float));
  float *weight_ref = (float*)malloc(oc*ic*fh*fw*sizeof(float));
  for (int i = 0; i < oc*ic*fh*fw; i++) {
    weight[i] = rand() % 5;
    //weight[i] = 1;
    weight_ref[i] = weight[i];
  }
  float *bias = (float*)malloc(oc*sizeof(float));
  float *bias_ref = (float*)malloc(oc*sizeof(float));
  for (int i = 0; i < oc; i++) {
    bias[i] = rand() % 5;
    //bias[i] = 0;
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

  // infer algorithm
  ConvAlg alg;
  infer_conv_alg(nb, ic/4, ih, iw, oc/4, oh, ow, fh, fw, s, p, &alg);

  // setup buffer
  int buf_bytes;
  conv_buffer_size(nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, alg, &buf_bytes);
  float *buf = (float*)malloc(buf_bytes);
  memset(buf, 0, buf_bytes);

  // transform weight
  int wtm_bytes;
  weight_trans_size(nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, alg, &wtm_bytes);
  float *wtm = (float*)malloc(wtm_bytes);
  memset(wtm, 0, wtm_bytes);
  
  weight_trans(weight, wtm, ic, oc, fh, fw, alg);

  // do conv
  double start, end;
  start = get_current_time();
  conv(input, wtm, output, bias, nb, ic, ih, iw, oc, oh, ow, fh, fw, s, p, buf, alg);
  end = get_current_time();
  std::cerr << end - start << " ms\n";

  // compare rst
  compare(output, output_ref, nb*oc*oh*ow);

  free(input);
  free(input_ref);
  free(weight);
  free(weight_ref);
  free(output);
  free(output_ref);
  free(bias);
  free(bias_ref);
  free(buf);
  free(wtm);
}