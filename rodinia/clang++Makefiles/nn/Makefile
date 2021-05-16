include ../../common/make.config

DPCPP_CXX = $(SYCL_ROOT)/bin/clang++
DPCPP_CXXFLAGS = -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS) -o
DPCPP_LDFLAGS = 
DPCPP_EXE_NAME = nn
DPCPP_SOURCES = nn.dp.cpp

RUN_OPTS = filelist.txt -r 5 -lat 30 -lng 90 -t

all:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)

build_dpcpp:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)



run:
	./$(DPCPP_EXE_NAME) $(RUN_OPTS)

run_dpcpp:
	./$(DPCPP_EXE_NAME)


clean: 
	rm -rf $(DPCPP_EXE_NAME) $(OMP_EXE_NAME)



