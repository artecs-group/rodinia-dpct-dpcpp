include ../../../common/make.config

CC = gcc
NVCC = $(SYCL_ROOT)/bin/clang++

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

NVCCFLAGS += -I../common \
			 -O3 \
			 -use_fast_math \
			 -lm \
			 -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS) \
			 $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.dp.cpp
# C/C++ source files (compiled with gcc / c++)
# CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.dp.o lud_kernel.dp.o

.PHONY: all clean 
all : $(EXECUTABLE)

common.o : common.c
	$(CC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o common.o -c common.c

lud.dp.o:	lud.dp.cpp 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o lud.dp.o -c lud.dp.cpp

lud_kernel.dp.o:	lud_kernel.dp.cpp 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o lud_kernel.dp.o -c lud_kernel.dp.cpp



$(EXECUTABLE) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@  $?

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
