include ../../common/make.config

CC := $(SYCL_ROOT)/bin/clang++

DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

SRC = bfs.dp.cpp 

EXE = bfs

release: $(SRC)
	$(CC) $(SRC) $(DPCPP_FLAGS) -o $(EXE) 

enum: $(SRC)
	$(CC) $(DPCPP_FLAGS) -deviceemu $(SRC) -o $(EXE) 

debug: $(SRC)
	$(CC) -g $(DPCPP_FLAGS) $(SRC) -o $(EXE) 
debugenum: $(SRC)
	$(CC) -g -deviceemu $(DPCPP_FLAGS) $(SRC) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
