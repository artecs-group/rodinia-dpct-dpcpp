
CC := dpcpp


SRC = bfs.dp.cpp 

EXE = bfs

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	DPCPP_FLAGS = -DTIME_IT
endif

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) $(DPCPP_FLAGS)

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) $(DPCPP_FLAGS)

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) $(DPCPP_FLAGS)
debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) $(DPCPP_FLAGS)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
