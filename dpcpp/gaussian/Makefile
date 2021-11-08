include ../../common/make.config

# DPCPP_CXX =  $(SYCL_ROOT)/bin/clang++
# DPCPP_CXXFLAGS = -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS) -o
DPCPP_LDFLAGS = 
DPCPP_EXE_NAME = gaussianElimination
DPCPP_SOURCES = gaussian.dp.cpp

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

RUN_OPTS = ../../data/gaussian/matrix4.txt

all:
	$(SYCL_C) $(DPCPP_FLAGS) -o $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)

run:
	./$(DPCPP_EXE_NAME) $(RUN_OPTS)


clean: 
	rm -rf $(DPCPP_EXE_NAME)



