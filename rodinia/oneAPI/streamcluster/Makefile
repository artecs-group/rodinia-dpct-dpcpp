include ../../common/make.config

NVCC = dpcpp

#NVCC_FLAGS = -I$(CUDA_DIR)/include

TARGET_G = sc_gpu


# make dbg=1 tells nvcc to add debugging symbols to the binary
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

# make emu=1 compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

# make dp=1 compiles the CUDA kernels with double-precision support
ifeq ($(dp),1)
	NVCC_FLAGS += --gpu-name sm_13
endif


$(TARGET_G): streamcluster_cuda_cpu.cpp.dp.cpp streamcluster_cuda.dp.cpp streamcluster_header.dp.cpp
	$(NVCC) $(NVCC_FLAGS) streamcluster_cuda_cpu.cpp.dp.cpp streamcluster_cuda.dp.cpp streamcluster_header.dp.cpp -o $(TARGET_G)
	

clean:
	rm -f *.o *~ *.txt $(TARGET_G) *.linkinfo
	
