
# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# SYCL compiler
DPCPP = dpcpp


# 'make dbg=1' enables DPCPP debugging
ifeq ($(dbg),1)
	DPCPP_FLAGS = -g -O0
else
	DPCPP_FLAGS = -O2
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
	$(DPCPP) $(DPCPP_FLAGS) backprop.o facetrain.o imagenet.o backprop_sycl.dp.o -o backprop 

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

facetrain.o: facetrain.c backprop.h
	$(CC) $(CC_FLAGS) facetrain.c -c
	
backprop.o: backprop.c backprop.h
	$(CC) $(CC_FLAGS) backprop.c -c

backprop_sycl.dp.o: backprop_sycl.dp.cpp backprop.h
	$(DPCPP) $(DPCPP_FLAGS) -c backprop_sycl.dp.cpp

imagenet.o: imagenet.c backprop.h
	$(CC) $(CC_FLAGS) imagenet.c -c


clean:
	rm -f *.o *~ backprop backprop_cuda.linkinfo
