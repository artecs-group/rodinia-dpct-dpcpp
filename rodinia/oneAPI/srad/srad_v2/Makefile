include ../../../common/make.config

CC := dpcpp

SRC = srad.dp.cpp

EXE = srad

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE)

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
