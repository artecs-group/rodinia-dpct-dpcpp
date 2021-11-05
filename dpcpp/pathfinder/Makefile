include ../../common/make.config

# SYCL_C := $(SYCL_ROOT)/bin/clang++

# DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

INCLUDE := 

SRC = pathfinder.dp.cpp

EXE = pathfinder

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif


release:
	$(SYCL_C) $(DPCPP_FLAGS) $(SRC) -o $(EXE)

enum:
	$(SYCL_C) $(DPCPP_FLAGS) -deviceemu $(SRC) -o $(EXE)

debug:
	$(SYCL_C) -g $(DPCPP_FLAGS) $(SRC) -o $(EXE)

debugenum:
	$(SYCL_C) -g $(DPCPP_FLAGS) -deviceemu $(SRC) -o $(EXE)

clean:
	rm -f pathfinder