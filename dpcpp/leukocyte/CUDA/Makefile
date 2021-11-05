include ../../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# C compiler
CC = gcc
CC_FLAGS = -g -O3 -Wall -I$(MATRIX_DIR)

# CUDA compiler
# SYCL_C = $(SYCL_ROOT)/bin/clang++
# DPCPP_FLAGS = -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# Matrix library path
MATRIX_DIR = ../meschach_lib

# 'make dbg=1' enables SYCL_C debugging
ifeq ($(dbg),1)
	DPCPP_FLAGS += -g -O0
else
	DPCPP_FLAGS += -O3
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	DPCPP_FLAGS += -deviceemu
endif


leukocyte: detect_main.o avilib.o find_ellipse.o find_ellipse_kernel.dp.o track_ellipse.o track_ellipse_kernel.dp.o misc_math.o $(MATRIX_DIR)/meschach.a
	$(SYCL_C) $(DPCPP_FLAGS) -lm -lsycl avilib.o find_ellipse.o find_ellipse_kernel.dp.o track_ellipse.o track_ellipse_kernel.dp.o misc_math.o detect_main.o -o leukocyte $(MATRIX_DIR)/meschach.a -lm

%.o: %.[ch]
	$(CC)  $(OUTPUT) $(CC_FLAGS) $< -c

detect_main.o: detect_main.c find_ellipse.h track_ellipse.h avilib.h 
	$(CC) $(CC_FLAGS) -I $(MATRIX_DIR) detect_main.c -c

find_ellipse_kernel.dp.o: find_ellipse_kernel.dp.cpp find_ellipse_kernel.h
	$(SYCL_C) $(DPCPP_FLAGS) -c find_ellipse_kernel.dp.cpp

track_ellipse_kernel.dp.o: track_ellipse_kernel.dp.cpp track_ellipse_kernel.h
	$(SYCL_C) $(DPCPP_FLAGS) -I$(MATRIX_DIR) -c track_ellipse_kernel.dp.cpp

find_ellipse.o: avilib.h
	$(CC) $(CC_FLAGS) -I$(MATRIX_DIR) find_ellipse.c -c

track_ellipse.o: find_ellipse.h avilib.h
	$(CC) $(CC_FLAGS) -I$(MATRIX_DIR) track_ellipse.c -c

avilib.o: avilib.h
	$(CC) $(CC_FLAGS) avilib.c -c

misc_math.o: avilib.h
	$(CC) $(CC_FLAGS) misc_math.c -c

$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); ./configure --with-all; make all; make clean

clean:
	rm -f *.o *~ leukocyte *.linkinfo *.gch
