void sgemm_8x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k);


void sgemm_4x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k);


void sgemm_1x8(float *in, float *wgt, float *out_0, float *out_1,
               float *b_0, float *b_1, int k);


void sgemm_8x4(float *in, float *wgt, float *out_0, float *b_0, int k);


void sgemm_4x4(float *in, float *wgt, float *out_0, float *b_0, int k);


void sgemm_1x4(float *in, float *wgt, float *out_0, float *b_0, int k);
