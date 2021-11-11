include ../../common/make.config

# SYCL_C = $(SYCL_ROOT)/bin/clang++

# DPCPP_FLAGS = -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

all: euler3d euler3d_double  pre_euler3d  pre_euler3d_double 

euler3d: euler3d.dp.cpp
	$(SYCL_C) $(KERNEL_DIM) -O3 $(DPCPP_FLAGS) euler3d.dp.cpp -o euler3d 

euler3d_double: euler3d_double.dp.cpp
	$(SYCL_C)  -O3 $(DPCPP_FLAGS) euler3d_double.dp.cpp -o euler3d_double 


pre_euler3d: pre_euler3d.dp.cpp
	$(SYCL_C)  -O3 $(DPCPP_FLAGS) pre_euler3d.dp.cpp -o pre_euler3d 

pre_euler3d_double: pre_euler3d_double.dp.cpp
	$(SYCL_C)  -O3 $(DPCPP_FLAGS) pre_euler3d_double.dp.cpp -o pre_euler3d_double 


clean:
	rm -f euler3d euler3d_double pre_euler3d pre_euler3d_double *.linkinfo
