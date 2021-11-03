include ../../common/make.config

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

C_C = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

CUD_C = $(SYCL_ROOT)/bin/clang++
# OMP_FLAG = 	-Xcompiler paste_one_here
CUDA_FLAG = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# link objects (binaries) together
a.out:		main.o \
			./kernel/kernel_gpu_cuda_wrapper.dp.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.dp.o
	$(CUD_C)	$(KERNEL_DIM) main.o \
			./kernel/kernel_gpu_cuda_wrapper.dp.o \
			./util/num/num.o \
			./util/timer/timer.o \
			./util/device/device.dp.o \
			-lm \
			$(OMP_LIB) $(CUDA_FLAG) \
			-o lavaMD

# compile function files into objects (binaries)
main.o:		main.hpp \
			main.cpp \
			./kernel/kernel_gpu_cuda_wrapper.h \
			./kernel/kernel_gpu_cuda_wrapper.dp.cpp \
			./util/num/num.h \
			./util/num/num.c \
			./util/timer/timer.h \
			./util/timer/timer.c \
			./util/device/device.h \
			./util/device/device.dp.cpp
	$(CUD_C)	$(KERNEL_DIM) $(OUTPUT) main.cpp \
			-c \
			-o main.o \
			-O3 $(CUDA_FLAG)

./kernel/kernel_gpu_cuda_wrapper.dp.o:	./kernel/kernel_gpu_cuda_wrapper.h \
									./kernel/kernel_gpu_cuda_wrapper.dp.cpp
	$(CUD_C) $(KERNEL_DIM)						./kernel/kernel_gpu_cuda_wrapper.dp.cpp \
									-c \
									-o ./kernel/kernel_gpu_cuda_wrapper.dp.o \
									-O3 \
									$(CUDA_FLAG)

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(C_C)				./util/timer/timer.c \
						-c \
						-o ./util/timer/timer.o \
						-O3

./util/device/device.dp.o:	./util/device/device.h \
						./util/device/device.dp.cpp
	$(CUD_C)			./util/device/device.dp.cpp \
						-c \
						-o ./util/device/device.dp.o \
						-O3 $(CUDA_FLAG)

# delete all object and executable files
clean:
	rm	*.o \
		./kernel/*.o \
		./util/num/*.o \
		./util/timer/*.o \
		./util/device/*.o \
		lavaMD
