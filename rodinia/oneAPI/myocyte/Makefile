# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
myocyte.out:		main.dp.o
	dpcpp	main.dp.o \
				-I/usr/local/cuda/include \
				-L/usr/local/cuda/lib \
				-lm \
                                -o myocyte.out

# compile main function file into object (binary)
main.dp.o:	define.c \
				main.dp.cpp \
				work.dp.cpp \
				solver.dp.cpp \
				embedded_fehlberg_7_8.dp.cpp \
				master.dp.cpp \
				kernel.dp.cpp \
				kernel_ecc.dp.cpp \
				kernel_cam.dp.cpp \
				kernel_fin.dp.cpp \
				work_2.dp.cpp \
				solver_2.dp.cpp \
				embedded_fehlberg_7_8_2.dp.cpp \
				kernel_2.dp.cpp \
				kernel_ecc_2.dp.cpp \
				kernel_cam_2.dp.cpp \
				kernel_fin_2.dp.cpp \
				file.c \
				timer.c
	dpcpp	main.dp.cpp \
				-c -O3 -g

# delete all object files
clean:
	rm *.o *.out output.txt
