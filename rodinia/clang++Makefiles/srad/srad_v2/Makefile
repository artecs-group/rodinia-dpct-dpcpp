include ../../../common/make.config

CC := $(SYCL_ROOT)/bin/clang++

CC_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

SRC = srad.dp.cpp

EXE = srad

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(CC_FLAGS)

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) $(CC_FLAGS)

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) $(CC_FLAGS)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) $(CC_FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
