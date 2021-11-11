# Rodinia Benchmarks for DPC++
Rodinia benchmarks for CUDA translated to DPC++ using the [Intel DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html#gs.g3bkj1). These benchmarks can run in CPU, and NVIDIA/Intel GPUs.

## Requirements
Before to run the benchmarks you have to install some dependencies.

* [CUDA Toolkit 11.4](https://developer.nvidia.com/cuda-11-4-0-download-archive) -> Mandatory to run in NVIDIA GPUs.
* [Intel LLVM open source compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) -> It will run the code on all the devices.
* [Intel Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html). The commercial "dpcpp" compiler is optional but it can cohabit with the previous compiler. However, it does not support NVIDIA GPUs.
* In order to run the "Hybridsort" test, you have to install some OpenGL libraries:
    ```
    sudo apt-get install freeglut3 freeglut3-dev
    sudo apt-get install binutils-goldc
    sudo apt-get install libglew1.5
    ```

Some additional configuration would be requiered (see Section [Known Issues](#known-issues)). 

## Project Configuration

## Compilation

## Known Issues
### DPCT Dependencies not Found 
vector_class not found: you should add it's definition (e.g. oneAPI). Add it in "sycl_workspace/llvm/sycl/include/CL/sycl/stl.hpp"

### OpenGL Dependency
You have to install OpenGL to run some tests:

```
sudo apt-get install freeglut3 freeglut3-dev
sudo apt-get install binutils-goldc
sudo apt-get install libglew1.5
``` 

