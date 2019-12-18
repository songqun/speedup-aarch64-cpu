# speedup-aarch64-cpu
A computing kernel implementation in ML inference framework aiming at theoretical limit on ARMv8 CPU in single thread. It will mainly aim at Cortex-A55 and Cortex-A76.

The basic supported data type is for <b>float32</b>. But it will support <b>float16</b> and <b>int8</b>. On Cortex-A55 and Cortex-A76, they support vec-inst <b>fmla</b> in float16 and <b>sdot/udot</b> in int8, not only in storage but also in computation. Thus the computation peak of float16 and int8 will be 2 and 4 times than float32 on Cortex-A55 and Cortex-A76.

The data layout is packed by vec-reg length in the whole project, which means <b>NCHWc4</b> in float32, <b>NCHWc8</b> in float16, and <b>NCHWc16</b> in int8.

The project will first implement <b>ConvOp</b> because ConvOp will occupy most time in inference (50% to 80%). ConvOp will be implemented in multiple ways/algorithms in order to optimize different sizes.

Now, there are many open-source inference frameworks such as NCNN, MNN, and TVM. By my profiler, these frameworks only reach about 50% peak on Cortex-A55, and 70% peak on Cortex-A76 in ConvOp in single thread. Also they do not support float16.

`The aim of this project is to exceed them and reach 70% to 80% on Cortex-A55, and 80% to 90% on Cortex-A76, and supports float16.`

The test platform is RedMi 7 Pro, with Qualcomm Snapdragon 675, which is almost the cheapest SoC with Cortex-A55 and Cortex-76 now...

The frequency of A55 is 1.66 GHz and A76 is 2.00 GHz.

## Benchmark
Benchmarks will be updated in *benchmark* folder.

It includes table comparing time between the project and other open-source inference framework in ConvOp. The size of ConvOp now is based on ResNet50. I will test more further like Squeezenet, GoogLenet, vgg...

After depthwise-ConvOp being implementd, the benchmark will add MobileNetV2.

The summary of the benchmark is shown below, based on <b>fp32</b>:

|  conv type  | ih | iw |  ic  |  oc  | theoretical time on A55 | my time  on A55 | percent % | NCNN time on A55 | MNN time on A55 |
|-------------|----|----|------|------|-------------------------|-----------------|-----------|------------------|-----------------|
| conv1x1s1p0 | 56 | 56 | 64   | 256  |          7.517          |      10.841     |     69    | 14.70            | 19.826          |
|             | 56 | 56 | 256  | 64   |          7.517          |      11.279     |     67    | 17.87            | 19.85           |
|             | 28 | 28 | 128  | 512  |          7.517          |      10.44      |     72    | 12.74            | 16.232          |
|             | 28 | 28 | 512  | 128  |          7.517          |      12.26      |     61    | 15.37            | 18.617          |
|             | 14 | 14 | 256  | 1024 |          7.517          |      9.887      |     76    | 12.97            | 16.304          |
|             | 14 | 14 | 1024 | 256  |          7.517          |      11.98      |     63    | 15.81            | 22.696          |
|             | 7  | 7  | 512  | 2048 |          7.517          |      11.54      |     65    | 13.33            | 19.045          |
|             | 7  | 7  | 2048 | 512  |          7.517          |      12.262     |     61    | 13.84            | 22.407          |
| conv3x3s1p1 | 56 | 56 | 64   | 64   |          16.913         |      15.17      |    111    | 23.50            | 18.736          |
|             | 28 | 28 | 128  | 128  |          16.913         |      11.509     |    147    | 12.89            | 23.006          |
|             | 14 | 14 | 256  | 256  |          16.913         |      13.816     |    122    | 20.67            | 22.703          |
|             | 7  | 7  | 512  | 512  |          16.913         |      15.25      |    111    | 40.13            | 30.391          |

I compare my kernel with popular used and fast inference framework on arm: NCNN and MNN. I'm faster on each size of ConvOp.

I will add comparison with tf-lite, pytorch-lite further. In 'some' common sense, these two are much slower than NCNN and MNN.

Paddle-lite is another very faster inference framework and I will add comparison with it further.

Note that conv1x1s1p0 uses **conv_im2col** (though in fact no need to do im2col, just GEMM) and conv3x3s1p1 uses **conv_wino**(winograd algorithm to reduce computation amount) with the same SGEMM kernel. The idea percent of conv_im2col is 70-80% and conv_wino is 120-150% on little core.

`Until now, I reach 60-70% of peak performance on little core (A55), close to my aim.`

The further optimization idea is to implement sgemm12x8 kernel using 32 vec-reg fully. I predict that I will reach 65-75% of peak performance after that.

## TODO

Urgent:
* Optimization on A76 and do comparison
* implement sgemm12x8 using 32 vec-reg fully.

Further:
* Add more detail descriptions how to write a high-performance kernel.
* Support other hardware like Vulkan, X86, CUDA...build the speedup world !
* Add float16, int8.
* Support multi-threading.


## How to compile

The project is based on <b>CMake</b> newer than 3.0. The simplest way to compile it is:

```
mkdir build
cd build/
cmake ..
make -j8
```

For cross-compiling onto aarch64, the simplest way is following [NCNN's way](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-arm-cortex-a-family-with-cross-compiling). The key is to download cross-compiler, write a crosscompiler.toolchain.cmake like NCNN's, and set:

`cmake -DCMAKE_TOOLCHAIN_FILE=${PATH}/crosscompiler.toolchain.cmake ..`


**float16 needs very high version of gcc and clang.**

## General Opimization Idea

* SIMD
* Packing
* Blocking
* Instruction Reorder

TODO: More description in detail.

## Convolution Algorithm

* im2col
* winograd
* direct (not implement)

TODO: More description in detail.

The header <b>conv.hpp</b> also includes some details.

## Trick

On Cortex-A55, q-form <b>fmla</b> and <b>ldr</b> cannot be dual issue, so instruction reorder cannot fully hide <b>ldr</b> with <b>fmla</b>.  
However, I find a trick to split q-form <b>ldr</b> into 3 instructions and all of them can be hidden by <b>fmla</b>.  
So we need a computing kernel with computing / load ratio > 3.

The kernel in this project is 


sgemm_8x8: 8\*K matrix multiples K\*8 matrix.  
Computing / load ratio is: (2(4\*1 weight)\*8(1 in)) / (2(4\*1 in) + 2(4\*1 weight)) = 4 > 3