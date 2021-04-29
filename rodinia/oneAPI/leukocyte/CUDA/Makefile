include ../../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# C compiler
CC = gcc
CC_FLAGS = -g -O3 -Wall -I$(MATRIX_DIR)

# CUDA compiler
NVCC = dpcpp
# NVCC_FLAGS = -I$(CUDA_DIR)/include -I$(SDK_DIR)/common/inc

# Matrix library path
MATRIX_DIR = ../meschach_lib

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif


leukocyte: detect_main.c.dp.o avilib.o find_ellipse.o find_ellipse_kernel.dp.o track_ellipse.o track_ellipse_kernel.dp.o misc_math.o $(MATRIX_DIR)/meschach.a
	$(NVCC) $(CC_FLAGS) -lm avilib.o find_ellipse.o find_ellipse_kernel.dp.o track_ellipse.o track_ellipse_kernel.dp.o misc_math.o detect_main.c.dp.o -o leukocyte $(MATRIX_DIR)/meschach.a -lm

%.o: %.[ch]
	$(CC)  $(OUTPUT) $(CC_FLAGS) $< -c

detect_main.c.dp.o: detect_main.c.dp.cpp find_ellipse.h track_ellipse.h avilib.h 
	$(NVCC) $(CC_FLAGS) detect_main.c.dp.cpp -c

find_ellipse_kernel.dp.o: find_ellipse_kernel.dp.cpp find_ellipse_kernel.h
	$(NVCC) $(NVCC_FLAGS) -c find_ellipse_kernel.dp.cpp

track_ellipse_kernel.dp.o: track_ellipse_kernel.dp.cpp track_ellipse_kernel.h
	$(NVCC) $(NVCC_FLAGS) -I$(MATRIX_DIR) -c track_ellipse_kernel.dp.cpp

find_ellipse.o: avilib.h

track_ellipse.o: find_ellipse.h avilib.h

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o *~ leukocyte *.linkinfo *.gch
