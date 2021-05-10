DPCPP_CXX = dpcpp
DPCPP_CXXFLAGS = -std=c++17 -g -o
DPCPP_LDFLAGS = 
DPCPP_EXE_NAME = gaussianElimination
DPCPP_SOURCES = gaussian.dp.cpp

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	DPCPP_LDFLAGS += -DTIME_IT
endif

RUN_OPTS = ../../data/gaussian/matrix4.txt

all:
	$(DPCPP_CXX) $(DPCPP_CXXFLAGS) $(DPCPP_EXE_NAME) $(DPCPP_SOURCES) $(DPCPP_LDFLAGS)

run:
	./$(DPCPP_EXE_NAME) $(RUN_OPTS)


clean: 
	rm -rf $(DPCPP_EXE_NAME)



