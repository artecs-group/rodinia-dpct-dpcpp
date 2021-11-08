include ../../../common/make.config

CC = gcc
# SYCL_C = $(SYCL_ROOT)/bin/clang++

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

DPCPP_FLAGS += -I../common \
			 -O3 \
			 -use_fast_math \
			 -lm \
			 $(SPACE)

# Add source files here
EXECUTABLE  := lud_cuda
# Cuda source files (compiled with cudacc)
CUFILES     := lud_kernel.dp.cpp
# C/C++ source files (compiled with gcc / c++)
# CCFILES     := lud.c lud_cuda.c ../common/common.c

OBJS = ../common/common.o lud.dp.o lud_kernel.dp.o

ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

.PHONY: all clean 
all : $(EXECUTABLE)

common.o : common.c
	$(CC) $(KERNEL_DIM) $(DPCPP_FLAGS) $(DEFS) -o common.o -c common.c

lud.dp.o:	lud.dp.cpp 
	$(SYCL_C) $(KERNEL_DIM) $(DPCPP_FLAGS) $(DEFS) -o lud.dp.o -c lud.dp.cpp

lud_kernel.dp.o:	lud_kernel.dp.cpp 
	$(SYCL_C) $(KERNEL_DIM) $(DPCPP_FLAGS) $(DEFS) -o lud_kernel.dp.o -c lud_kernel.dp.cpp



$(EXECUTABLE) : $(OBJS)
	$(SYCL_C) $(DPCPP_FLAGS) -o $@  $?

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
