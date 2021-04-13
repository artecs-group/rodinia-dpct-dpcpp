
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# link objects(binaries) together
heartwall: main.dp.o ./AVI/avilib.o ./AVI/avimod.o 
	dpcpp main.dp.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall

# compile main function file into object (binary)
main.dp.o: main.dp.cpp kernel.dp.cpp define.c
	dpcpp $(OUTPUT) $(KERNEL_DIM) main.dp.cpp -I./AVI -c -O3

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
