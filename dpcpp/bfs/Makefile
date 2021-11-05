include ../../common/make.config

# SYCL_C := $(SYCL_ROOT)/bin/clang++

# DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

SRC = bfs.dp.cpp 

EXE = bfs

release: $(SRC)
	$(SYCL_C) $(SRC) $(DPCPP_FLAGS) -o $(EXE) 

enum: $(SRC)
	$(SYCL_C) $(DPCPP_FLAGS) -deviceemu $(SRC) -o $(EXE) 

debug: $(SRC)
	$(SYCL_C) $(DPCPP_FLAGS) $(SRC) -o $(EXE) 
debugenum: $(SRC)
	$(SYCL_C) -deviceemu $(DPCPP_FLAGS) $(SRC) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
