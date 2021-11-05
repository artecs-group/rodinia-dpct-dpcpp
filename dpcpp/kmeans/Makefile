include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -O2 -fopenmp

# CUDA compiler
SYCL_C = $(SYCL_ROOT)/bin/clang++
# DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# 'make dbg=1' enables SYCL_C debugging
ifeq ($(dbg),1)
	DPCPP_FLAGS += -g -O0
else
	DPCPP_FLAGS += -O2
endif


kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.dp.o rmse.o
	$(SYCL_C) $(DPCPP_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.dp.o rmse.o -o kmeans -fopenmp -lm

kmeans.o: kmeans.c kmeans.h
	$(CC) $(CC_FLAGS) -c kmeans.c kmeans.h

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

kmeans_cuda.dp.o: kmeans_cuda.dp.cpp 
	$(SYCL_C) $(DPCPP_FLAGS) -fopenmp -c kmeans_cuda.dp.cpp

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo
