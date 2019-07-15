enum ConvAlg
{
  CONV_1X1S1P0,
  CONV_WINO_TWO_STEP,
  CONV_WINO_ONE_STEP,
  CONV_IM2COL,
  CONV_NOT_MATCH,
};


/*
 * nb: num_batch, ic: input_channel, ih: input_height, iw: input_width, oc: output_channel, s: stride, p: padding, oh: output_height, ow: output_width
 * input data layout: NCHWc4
 * output data layout: NOHWo4
 * weight data layout: it depends
 * computing kernel: sgemm_8x8
 * Assume ic and oc have been divided by 4
 */
void conv(const float *input, const float *weight, float *output, const float *bias,
          int nb, int ic, int ih, int iw, int oc, int oh, int ow,
          int fh, int fw, int s, int p, float *buf, ConvAlg alg);


/*
 * fh=fw=1, s=1, p=0. It is a GEMM
 * do input packing when first meet in computing. NCHWhw8c4 => NHWCc4hw8 (NCHWc4 => NHWChw8)
 * weight data layout: OCo8 (fh=fw=1)
 * Assume ic and oc have been divided by 4
 */
void conv1x1s1p0(const float *input, const float *weight, float *output, const float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow,
                 int fh, int fw, int s, int p, float *buf);


/*
 * winograd, F(4x4, 6x6)
 * two step: store the whole in_tm, then do dot prod, then transform the whole out_tm
 * input trans data layout: NCHWc4 => N*(6*6)*HW*C*hw8
 * output trans data layout: N*(6*6)*O*HW*o8 => NOHWo4
 * weight data layout: N*(6*6)*O*T*C*o8 (T = oh/4*ow/4, tiles: nb of 6x6 block on one channel)
 * Assume ic and oc have been divided by 4
 */
void conv3x3s1p1_wino_two_step(const float *input, const float *weight, float *output, const float *bias,
                               int nb, int ic, int ih, int iw, int oc, int oh, int ow,
                               int fh, int fw, int s, int p, float *buf);


/*
 * winograd, F(4x4, 6x6)
 * one step: do partly in_tm, then do dot prod on them, and transform them back
 * maybe better than two step for all data loaded into cache only once if blocked properly
 * input trans data layout: NCHWc4 => N*(6*6)*HW*C*hw8
 * output trans data layout: N*(6*6)*O*HW*o8 => NOHWo4
 * weight data layout: N*(6*6)*O*T*C*o8 (T = oh/4*ow/4, tiles: num of 6x6 block on one channel)
 * Assume ic and oc have been divided by 4
 */
void conv3x3s1p1_wino_one_step(const float *input, const float *weight, float *output, const float *bias,
                               int nb, int ic, int ih, int iw, int oc, int oh, int ow,
                               int fh, int fw, int s, int p, float *buf);


/*
 * im2col + GEMM, fit general case
 * only open 8*fh*fw*C buffer to do im2col when first meet in computing instead of HW*fh*fw*C for whole input
 * do input packing when first meet. NCHWhw8c4 => NHWCc4hw8 (NCHWc4 => NHWChw8)
 * weight data layout: OHWCo8
 * Assume ic and oc have been divided by 4
 */
void conv_im2col(const float *input, const float *weight, float *output, const float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow,
                 int fh, int fw, int s, int p, float *buf);


/*
 * maybe not implement
 * weight data layout: OCHWo8
 * Assume ic and oc have been divided by 4
 */
void conv_direct(const float *input, const float *weight, float *output, const float *bias,
                 int nb, int ic, int ih, int iw, int oc, int oh, int ow,
                 int fh, int fw, int s, int p, float *buf);


/*
 * get oh and ow
 */
void get_output_hw(int ih, int iw, int fh, int fw, int s, int p, int *oh, int *ow);


/*
 * transform weight from OCHWo8 into corresponding conv computing
 * no optimization because the weight with proper layout can be stored into model, 
 * and next time when we load the model, the weight has already been in proper layout.
 */
void weight_trans(const float *weight, float *weight_tm);


/*
 * weight data layout: OCo8 (fh=fw=1)
 */
void weight_trans_1x1s1p0(const float *weight, float *weight_tm);


/*
 * weight data layout: OHWCo8
 */
void weight_trans_im2col(const float *weight, float *weight_tm);


/*
 * maybe not implement
 * weight data layout: OCHWo8
 */
void weight_trans_direct(const float *weight, float *weight_tm);


/*
 * winograd, F(4x4, 6x6)
 * weight data layout: N*(6*6)*O*T*C*o8 (T = oh/4*ow/4, tiles: num of 6x6 block on one channel)
 */
void weight_trans_wino(const float *weight, float *weight_tm);


/*
 * winograd, F(4x4, 6x6)
 * input trans data layout: NCHWc4 => N*(6*6)*HW*C*hw8
 */
void input_trans_wino(const float *input, float *input_tm);


/*
 * winograd, F(4x4, 6x6)
 * output trans data layout: N*(6*6)*O*HW*o8 => NOHWo4
 */
void output_trans_wino(const float *output, float *output_tm);


/*
 * infer conv algorithm
 * Assume ic and oc have been divided by 4
 */
void infer_conv_alg(int nb, int ic, int ih, int iw, int oh, int ow, int oc,
                    int fh, int fw, int s, int p, ConvAlg *alg);


/*
 * compute buffer size for specific conv algorithm
 * Assume ic and oc have been divided by 4
 */
void conv_buffer_size(int nb, int ic, int ih, int iw, int oh, int ow, int oc, int fh, int fw, int s, int p, ConvAlg alg, int *bytes);
