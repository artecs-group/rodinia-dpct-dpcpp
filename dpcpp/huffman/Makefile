include ../../common/make.config

# SYCL_C = $(SYCL_ROOT)/bin/clang++
CC = g++

DPCPP_FLAGS = -O3 -m64 $(DPCPP_FLAGS)

GCC_OPTS=-O3 -Wall -Wextra -m64 -I $(SYCL_INC)

OBJ = main_test_cu.dp.o stats_logger.o vlc_kernel_sm64huff.dp.o scan.dp.o pack_kernels.dp.o cpuencode.dp.o
SRC = main_test_cu.dp.cpp
EXE = pavle

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

pavle: $(OBJ) 
	$(SYCL_C) $(TESTING) $(CACHECWLUT) $(DPCPP_FLAGS) $(OBJ) -o $(EXE) 

vlc_kernel_sm64huff.dp.o: vlc_kernel_sm64huff.dp.cpp 
	$(SYCL_C) $(TESTING) -c vlc_kernel_sm64huff.dp.cpp $(DPCPP_FLAGS)

scan.dp.o: scan.dp.cpp 
	$(SYCL_C) -c scan.dp.cpp $(DPCPP_FLAGS)

cpuencode.dp.o: cpuencode.dp.cpp
	$(SYCL_C) -c cpuencode.dp.cpp $(DPCPP_FLAGS)

pack_kernels.dp.o: pack_kernels.dp.cpp 
	$(SYCL_C) -c pack_kernels.dp.cpp $(DPCPP_FLAGS)

main_test_cu.dp.o: main_test_cu.dp.cpp cutil.h
	$(SYCL_C) $(TESTING) -c main_test_cu.dp.cpp $(DPCPP_FLAGS)

.o:.cpp
	$(CC) ++ $(GCC_OPTS) -c $@ -o $<

clean:
	rm -f *.o $(EXE) 
