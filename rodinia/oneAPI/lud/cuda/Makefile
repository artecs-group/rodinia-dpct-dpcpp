CC = gcc
NVCC = dpcpp

DEFS += \
		-DGPU_TIMER \
		$(SPACE)

NVCCFLAGS += -I../common \
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

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	TIMER = -DTIME_IT
endif

.PHONY: all clean 
all : $(EXECUTABLE)

common.o : common.c
	$(CC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o common.o -c common.c 

lud.dp.o:	lud.dp.cpp 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o lud.dp.o -c lud.dp.cpp $(TIMER)

lud_kernel.dp.o:	lud_kernel.dp.cpp 
	$(NVCC) $(KERNEL_DIM) $(NVCCFLAGS) $(DEFS) -o lud_kernel.dp.o -c lud_kernel.dp.cpp $(TIMER)



$(EXECUTABLE) : $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(TIMER) -o $@  $?

clean:
	rm -f $(EXECUTABLE) $(OBJS) *.linkinfo
