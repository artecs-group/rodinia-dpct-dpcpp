include ../../common/make.config

CC := dpcpp

INCLUDE := 

SRC = pathfinder.dp.cpp

EXE = pathfinder

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	FLAGS = -DTIME_IT
endif

release:
	$(CC) $(SRC) -o $(EXE) $(FLAGS)

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) $(FLAGS)

debug:
	$(CC) -g $(SRC) -o $(EXE) $(FLAGS)

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) $(FLAGS)

clean:
	rm -f pathfinder