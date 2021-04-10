
all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double 

euler3d: euler3d.dp.cpp
	dpcpp $(KERNEL_DIM) -O2  euler3d.dp.cpp -o euler3d 

euler3d_double: euler3d_double.dp.cpp
	dpcpp  -O3 euler3d_double.dp.cpp -o euler3d_double 


pre_euler3d: pre_euler3d.dp.cpp
	dpcpp  -O3 pre_euler3d.dp.cpp -o pre_euler3d 

pre_euler3d_double: pre_euler3d_double.dp.cpp
	dpcpp  -O3 pre_euler3d_double.dp.cpp -o pre_euler3d_double 


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
