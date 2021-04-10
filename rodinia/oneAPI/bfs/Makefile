
CC := dpcpp


SRC = bfs.dp.cpp 

EXE = bfs

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) 

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) 
debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
