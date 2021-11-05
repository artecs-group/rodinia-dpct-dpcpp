
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifeq ($(time),1)
	FLAGS = -DTIME_IT
endif

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o 
	nvcc main.o ./AVI/avilib.o ./AVI/avimod.o -I/usr/local/cuda/include -lm -o heartwall

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3 $(FLAGS)

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
