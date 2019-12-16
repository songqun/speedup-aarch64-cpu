# speedup-aarch64-cpu
A computing kernel implementation in ML inference framework aiming at theoretical limit on ARMv8 CPU in single thread. It will mainly aim at Cortex-A55 and Cortex-A76.

The basic supported data type is for <b>float32</b>. But it will support <b>float16</b> and <b>int8</b>. On Cortex-A55 and Cortex-A76, they support vec-inst <b>fmla</b> in float16 and <b>sdot/udot</b> in int8, not only in storage but also in computation. Thus the computation peak of float16 and int8 will be 2 and 4 times than float32 on Cortex-A55 and Cortex-A76.

The data layout is packed by veg-reg length in the whole project, which means <b>NCHWc4</b> in float32, <b>NCHWc8</b> in float16, and <b>NCHWc16</b> in int8.

The project will first implement <b>ConvOp</b> because ConvOp will occupy most time in inference (50% to 80%). ConvOp will be implemented in multiple ways/algorithms in order to optimize different sizes.

Now, there are many open-source inference frameworks such as NCNN, MNN, and TVM. By my profiler, these frameworks only reach about 50% peak on Cortex-A55, and 70% peak on Cortex-A76 in ConvOp in single thread. Also they do not support float16.

`The aim of this project is to exceed them and reach 70% to 80% on Cortex-A55, and 80% to 90% on Cortex-A76, and supports float16.`

The test platform is RedMi 7 Pro, with Qualcomm Snapdragon 675, which is almost the cheapest SoC with Cortex-A55 and Cortex-76 now...

The frequency of A55 is 1.66 GHz and A76 is 2.00 GHz.

## Benchmark
Benchmarks will be updated in *benchmark* folder.

It will include tables comparing FLOPS and time between the project and other open-source inference framework in ConvOp firstly. The size of ConvOp will be based on GoogLeNet and ResNet50.

After depthwise-ConvOp being implementd, the benchmark will add MobileNetV2.

## TODO

* Add more Operators and support End-to-End benchmarks.
* Add float16, int8.
* Design own model serialization format and add model transform tools.
* Support multi-threading.
* Support other hardware like Vulkan, X86, CUDA...


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

<!--
## Trick

On Cortex-A55, q-form <b>fmla</b> and <b>ldr</b> cannot be dual issue, so instruction reorder cannot fully hide <b>ldr</b> with <b>fmla</b>.  
However, I find a trick to split q-form <b>ldr</b> into 3 instructions and all of them can be hidden by <b>fmla</b>.  
So we need a computing kernel with computing / load ratio > 3.

The kernel in this project is 

sgemm_4x16: 4\*K matrix multiples K\*16 matrix.  
Computing / load ratio is: (4(4\*1 weight)\*4(1 in)) / (1(4\*1 in) + 4(4\*1 weight)) = 16/5 > 3.

Or 

sgemm_8x8: 8\*K matrix multiples K\*8 matrix.  
Computing / load ratio is: (2(4\*1 weight)\*8(1 in)) / (2(4\*1 in) + 2(4\*1 weight)) = 4 > 3
-->