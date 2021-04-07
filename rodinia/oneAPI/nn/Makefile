DPCPP_CXX = dpcpp
DPCPP_CXXFLAGS = -std=c++17 -g -o
DPCPP_LDFLAGS = 
DPCPP_EXE_NAME = nn_dpcpp
DPCPP_SOURCES = nn_dpcpp.cpp

RUN_OPTS = filelist.txt -r 5 -lat 30 -lng 90

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



