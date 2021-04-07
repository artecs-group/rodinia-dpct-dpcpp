include ../../common/make.config

CC := dpcpp

INCLUDE := 

SRC = pathfinder.dp.cpp

EXE = pathfinder

release:
	$(CC) $(SRC) -o $(EXE)

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE)

debug:
	$(CC) -g $(SRC) -o $(EXE)

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE)

clean:
	rm -f pathfinder