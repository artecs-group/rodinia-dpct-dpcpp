include ../../common/make.config

CC := $(SYCL_ROOT)/bin/clang++

SRC = 3D.dp.cpp

EXE = 3D 

OUTPUT = *.out

FLAGS = -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

ifeq ($(time),1)
	FLAGS += -DTIME_IT
endif

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) $(SRC) -o $(EXE)  

enum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -deviceemu $(SRC) -o $(EXE)

debug: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g $(SRC) -o $(EXE)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) $(FLAGS) -g -deviceemu $(SRC) -o $(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo $(OUTPUT) 
