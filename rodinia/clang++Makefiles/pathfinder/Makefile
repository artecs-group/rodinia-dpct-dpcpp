include ../../common/make.config

CC := $(SYCL_ROOT)/bin/clang++

CC_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

INCLUDE := 

SRC = pathfinder.dp.cpp

EXE = pathfinder

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	FLAGS = -DTIME_IT
endif


release:
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE)

enum:
	$(CC) $(CC_FLAGS) -deviceemu $(SRC) -o $(EXE)

debug:
	$(CC) -g $(CC_FLAGS) $(SRC) -o $(EXE)

debugenum:
	$(CC) -g $(CC_FLAGS) -deviceemu $(SRC) -o $(EXE)

clean:
	rm -f pathfinder