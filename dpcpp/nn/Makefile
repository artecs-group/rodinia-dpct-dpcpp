include ../../common/make.config

# SYCL_C = $(SYCL_ROOT)/bin/clang++
DPCPP_FLAGS = -g-o $(DPCPP_FLAGS)
DPCPP_LDFLAGS = 
DPCPP_EXE_NAME = nn
DPCPP_SOURCES = nn.dp.cpp

RUN_OPTS = filelist.txt -r 5 -lat 30 -lng 90 -t

all:
	$(SYCL_C) $(DPCPP_FLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)

build_dpcpp:
	$(SYCL_C) $(DPCPP_FLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)



run:
	./$(DPCPP_EXE_NAME) $(RUN_OPTS)

run_dpcpp:
	./$(DPCPP_EXE_NAME)


clean: 
	rm -rf $(DPCPP_EXE_NAME) $(OMP_EXE_NAME)



