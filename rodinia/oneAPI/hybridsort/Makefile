include ../../common/make.config
CC := dpcpp

#CC_FLAGS = -arch=sm_20

ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

hybridsort: main.dp.cpp  bucketsort.dp.cpp bucketsort.dp.hpp bucketsort_kernel.dp.cpp histogram1024_kernel.dp.cpp  mergesort.dp.cpp mergesort.dp.hpp mergesort_kernel.dp.cpp
	$(CC) -g $(CC_FLAGS) $(VERIFY) $(OUTPUT) $(TIMER) $(HISTO_WG_SIZE_0) $(BUCKET_WG_SIZE_0) $(BUCKET_WG_SIZE_1) $(MERGE_WG_SIZE_0) $(MERGE_WG_SIZE_1) bucketsort.dp.cpp mergesort.dp.cpp main.dp.cpp -o hybridsort


clean:
	rm	*.o hybridsort
