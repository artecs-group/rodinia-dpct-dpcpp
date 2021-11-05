#makefile
include ../../common/make.config

# SYCL_C := $(SYCL_ROOT)/bin/clang++
# DPCPP_FLAGS = -std=c++17 -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

all: naive float

naive: ex_particle_SYCL_naive_seq.dp.cpp
	$(SYCL_C) -g -lm -O3 $(DPCPP_FLAGS) -use_fast_math ex_particle_SYCL_naive_seq.dp.cpp -o particlefilter_naive

float: ex_particle_SYCL_float_seq.dp.cpp
	$(SYCL_C) -g -lm $(DPCPP_FLAGS) -use_fast_math ex_particle_SYCL_float_seq.dp.cpp -o particlefilter_float

clean:
	rm particlefilter_naive particlefilter_float
