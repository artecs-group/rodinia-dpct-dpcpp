include ../../common/make.config

SYCLCC=dpcpp
CC =g++

SYCLCC_OPTS=-O3 -m64 -g 

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
	$(SYCLCC) $(TESTING) $(CACHECWLUT) $(SYCLCC_OPTS) $(OBJ) -o $(EXE) 

vlc_kernel_sm64huff.dp.o: vlc_kernel_sm64huff.dp.cpp 
	$(SYCLCC) $(TESTING) -c vlc_kernel_sm64huff.dp.cpp $(SYCLCC_OPTS)

scan.dp.o: scan.dp.cpp 
	$(SYCLCC) -c scan.dp.cpp $(SYCLCC_OPTS)

cpuencode.dp.o: cpuencode.dp.cpp
	$(SYCLCC) -c cpuencode.dp.cpp $(SYCLCC_OPTS)

pack_kernels.dp.o: pack_kernels.dp.cpp 
	$(SYCLCC) -c pack_kernels.dp.cpp $(SYCLCC_OPTS)

main_test_cu.dp.o: main_test_cu.dp.cpp cutil.h
	$(SYCLCC) $(TESTING) -c main_test_cu.dp.cpp $(SYCLCC_OPTS)

.o:.cpp
	$(CC) ++ $(GCC_OPTS) -c $@ -o $<

clean:
	rm -f *.o $(EXE) 
