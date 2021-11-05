include ../../common/make.config

# CC := $(SYCL_ROOT)/bin/clang++

SRC = 3D.dp.cpp

EXE = 3D 

OUTPUT = *.out

# DPCPP_FLAGS = -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_DPCPP_FLAGS)

ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

release: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) $(DPCPP_FLAGS) $(SRC) -o $(EXE)  

enum: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) $(DPCPP_FLAGS) -deviceemu $(SRC) -o $(EXE)

debug: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) $(DPCPP_FLAGS) -g $(SRC) -o $(EXE)

debugenum: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) $(DPCPP_FLAGS) -g -deviceemu $(SRC) -o $(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
