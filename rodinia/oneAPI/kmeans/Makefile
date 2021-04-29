include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g 
#-O2 

# CUDA compiler
NVCC = dpcpp
#NVCC_FLAGS = -I$(CUDA_DIR)/include 

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	#NVCC_FLAGS += -g -O0
else
	#NVCC_FLAGS += -O2
endif


kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.dp.o rmse.o
	$(NVCC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.dp.o rmse.o -o kmeans -lm

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

kmeans_cuda.dp.o: kmeans_cuda.dp.cpp 
	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.dp.cpp

clean:
	rm -f *.o *~ kmeans kmeans_cuda.linkinfo
