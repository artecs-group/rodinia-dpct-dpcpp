include ../../common/make.config

CC := dpcpp

#INCLUDE := $(CUDA_DIR)/include

SRC = 3D.dp.cpp

EXE = 3D 

OUTPUT = *.out

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	FLAGS = -DTIME_IT
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
