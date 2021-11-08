include ../../common/make.config

# SYCL_C = $(SYCL_ROOT)/bin/clang++

# DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

TARGET_G = sc_gpu


# make dbg=1 tells SYCL_C to add debugging symbols to the binary
ifeq ($(dbg),1)
	DPCPP_FLAGS += -g -O0
else
	DPCPP_FLAGS += -O3
endif

# make emu=1 compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	DPCPP_FLAGS += -deviceemu
endif

# make dp=1 compiles the CUDA kernels with double-precision support
ifeq ($(dp),1)
	DPCPP_FLAGS += --gpu-name sm_13
endif

# make dp=1 compiles the CUDA kernels with double-precision support
ifeq ($(dp),1)
	DPCPP_FLAGS += --gpu-name sm_13
endif

$(TARGET_G): streamcluster_cuda_cpu.cpp.dp.cpp streamcluster_cuda.dp.cpp streamcluster_header.dp.cpp
	$(SYCL_C) $(DPCPP_FLAGS) streamcluster_cuda_cpu.cpp.dp.cpp streamcluster_cuda.dp.cpp streamcluster_header.dp.cpp -o $(TARGET_G)
	

clean:
	rm -f *.o *~ *.txt $(TARGET_G) *.linkinfo
	
