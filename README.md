# Rodinia Benchmarks for DPC++
Rodinia benchmarks for CUDA translated to DPC++ using the [Intel DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html#gs.g3bkj1). These benchmarks can run in CPU, and NVIDIA/Intel GPUs.

## Requirements
Before to run the benchmarks you have to install some dependencies.

* [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-11-4-0-download-archive) &#8594; Mandatory to run in NVIDIA GPUs.
* [Intel LLVM open source compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) &#8594; It will run the code on all the devices.
* [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html). The commercial "dpcpp" compiler cohabits with the previous compiler. However, it does not support NVIDIA GPUs.
* In order to run the "Hybridsort" test, you have to install some OpenGL libraries:
    ```
    sudo apt-get install freeglut3 freeglut3-dev
    sudo apt-get install binutils-goldc
    sudo apt-get install libglew1.5
    ```

Some additional configuration would be requiered (see Section [Known Issues](#known-issues)). 

## Project Configuration
Once you installed all the requirements, you have to edit the file "common/make.config", changing the value of the following variables:

* CUDA_DIR &#8594; set the path where you have installed the CUDA Toolkit (e.g. /usr/local/cuda)
* LLVM_DIR &#8594; set the location where you have installed the LLVM compiler "bin/build" folder (e.g. ~/sycl_workspace/llvm/build)
* ONEAPI_DIR &#8594; set the location where you have installed the oneAPI Base Toolkit (e.g. /opt/intel/oneapi)

## Compilation
At this point, there are two Makefiles to build the tests, one of them for CUDA tests and another for DPC++ tests.

### CUDA Tests Compilation
Move to cuda folder and invoke the make command with the following arguments:

* time=<0,1> &#8594; Prints the time consumption of the device.

Example:
    ```
    cd cuda
    make time=1
    ```

### DPC++ Tests Compilation
Move to dpcpp folder and invoke the make command with the following arguments:

* time=<0,1> &#8594; Prints the time consumption of the device.
* DPCPP_ENV=<clang,oneapi> &#8594; The "clang" option uses the LLVM compiler, while the "oneapi" uses the oneAPI compiler.
* DEVICE=<CPU,INTEL_GPU,NVIDIA_GPU> &#8594; Selects the device where the code runs. The "NVIDIA_GPU" option just works selecting the variable "DPCPP_ENV=clang".

The following example compiles the tests using the LLVM compiler, selects the NVIDIA GPU, and choose to show the GPU time consumption:
    ```
    cd dpcpp
    make DPCPP_ENV=clang DEVICE=NVIDIA_GPU time=1
    ```

## How to run it?
You can run them one by one, or use the scripts we provide ("time_cuda.sh", "time_dpcpp.sh"), which save the kernel time in a "timing" folder. For that, you had to compile them with the "time=1" argument.

## Known Issues
### DPCT Dependencies not Found
If compiling tests with LLVM, errors pop up (e.g. "vector_class not found") then probably the LLVM dependencies do not fit with all the DPCT library.
To fix it you have to copy the content of the file "path/to/oneapi/compiler/latest/linux/include/sycl/CL/sycl/stl.hpp" from oneAPI compiler to the the file "path/to/llvm/sycl/include/CL/sycl/stl.hpp" to the LLVM compiler. Now you have to [rebuild the LLVM compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain). 

### Tests not working in DPC++
The following tests does not work in DPC++:

* Hybridsort
* Kmeans
* Leukocyte
* Mummergpu

## Acknowledgements
This work has been supported by the EU (FEDER) and the Spanish MINECO and CM under grants S2018/TCS-4423 and RTI2018-093684-B-I00.