## Issues
### DPCT compatibility issue 
vector_class does not found: you should add it's definition (e.g. oneAPI) and add it in "sycl_workspace/llvm/sycl/include/CL/sycl/stl.hpp"

### OpenGL dependency
You have to install OpenGL to run some tests:

```
sudo apt-get install freeglut3 freeglut3-dev
sudo apt-get install binutils-goldc
sudo apt-get install libglew1.5
``` 

