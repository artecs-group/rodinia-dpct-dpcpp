include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = hotspot

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	FLAGS += -DTIME_IT
endif

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  $(FLAGS)

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  $(FLAGS)

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  $(FLAGS)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  $(FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
