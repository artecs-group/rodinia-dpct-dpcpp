include ../../common/make.config

CUDA_SDK_PATH := $(SDK_DIR)

# Determine the correct version of the cutil library
CUTIL_LIB = # -lcutil
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -e $(SDK_DIR)/lib/libcutil_x86_64.a; then echo T; else echo F; fi), T)
        CUTIL_LIB = #-lcutil_x86_64
     endif
endif

FLAGS = -arch sm_61

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	FLAGS += -DTIME_IT
endif

all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double 

euler3d: euler3d.cu
	nvcc $(KERNEL_DIM) -O2 -v euler3d.cu -o euler3d $(FLAGS)

euler3d_double: euler3d_double.cu
	nvcc -Xptxas -v -O3 euler3d_double.cu -o euler3d_double $(FLAGS)


pre_euler3d: pre_euler3d.cu
	nvcc -Xptxas -v -O3 pre_euler3d.cu -o pre_euler3d $(FLAGS)

pre_euler3d_double: pre_euler3d_double.cu
	nvcc -Xptxas -v -O3 pre_euler3d_double.cu -o pre_euler3d_double $(FLAGS)


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
