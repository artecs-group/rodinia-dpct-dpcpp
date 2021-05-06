include ../../common/make.config

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

DPCPP = $(SYCL_ROOT)/bin/clang++
DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)
# link objects(binaries) together
heartwall: main.dp.o ./AVI/avilib.o ./AVI/avimod.o 
	$(DPCPP) main.dp.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall $(DPCPP_FLAGS)

# compile main function file into object (binary)
main.dp.o: main.dp.cpp kernel.dp.cpp define.c
	$(DPCPP) $(OUTPUT) $(KERNEL_DIM) main.dp.cpp -I./AVI -c -O3 $(DPCPP_FLAGS)

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
