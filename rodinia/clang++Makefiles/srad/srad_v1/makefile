include ../../../common/make.config

CC := $(SYCL_ROOT)/bin/clang++

CC_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
a.out:		main.dp.o
	$(CC) $(CC_FLAGS)	main.dp.o \
				-lm -o srad

# compile main function file into object (binary)
main.dp.o: 	main.dp.cpp \
				define.c \
				graphics.c \
				extract_kernel.dp.cpp \
				prepare_kernel.dp.cpp \
				reduce_kernel.dp.cpp \
				srad_kernel.dp.cpp \
				srad2_kernel.dp.cpp \
				compress_kernel.dp.cpp
	$(CC) $(CC_FLAGS)	main.dp.cpp \
				-c -O3

# delete all object files
clean:
	rm *.o srad
