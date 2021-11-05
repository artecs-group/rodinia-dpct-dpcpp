include ../../common/make.config

# CC := $(SYCL_ROOT)/bin/clang++

# DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

SRC = hotspot.dp.cpp

EXE = hotspot

ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

release: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) $(SRC) -o $(EXE) $(DPCPP_FLAGS)

enum: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) $(DPCPP_FLAGS)

debug: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) -g $(SRC) -o $(EXE) $(DPCPP_FLAGS)

debugenum: $(SRC)
	$(SYCL_C) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) $(DPCPP_FLAGS)
clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
