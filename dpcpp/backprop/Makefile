include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# SYCL compiler
# SYCL_C = $(SYCL_ROOT)/bin/clang++
# DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# 'make dbg=1' enables DPCPP debugging
ifeq ($(dbg),1)
	DPCPP_FLAGS += -g -O0
else
	DPCPP_FLAGS = -O2 $(DPCPP_FLAGS)
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	DPCPP_FLAGS += -deviceemu
endif
# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

backprop: backprop.o facetrain.o imagenet.o backprop_sycl.dp.o 
	$(SYCL_C) $(DPCPP_FLAGS) backprop.o facetrain.o imagenet.o backprop_sycl.dp.o -o backprop 

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_sycl.dp.o: backprop_sycl.dp.cpp backprop.h
	$(SYCL_C) $(DPCPP_FLAGS) -c backprop_sycl.dp.cpp

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
