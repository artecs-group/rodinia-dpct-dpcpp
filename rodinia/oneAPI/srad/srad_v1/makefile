include ../../../common/make.config

CC := dpcpp


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
	$(CC)	main.dp.o \
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
	$(CC)	main.dp.cpp \
				-c -O3

# delete all object files
clean:
	rm *.o srad
