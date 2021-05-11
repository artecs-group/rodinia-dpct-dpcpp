include ../../common/make.config

CC := dpcpp

INCLUDE := 

SRC = hotspot.dp.cpp

EXE = hotspot

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	FLAGS += -DTIME_IT
endif

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) $(FLAGS)

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) $(FLAGS)

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) $(FLAGS)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) $(FLAGS)
clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
