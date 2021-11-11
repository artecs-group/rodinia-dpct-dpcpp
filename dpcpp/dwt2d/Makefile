#Configure
EXECUTABLE := dwt2d

# Debug options
dbg=0
cudagdb=0 #compile for use with cuda-gdb, note that 'dbg' must be 1 as well
performancetest=1

# NVCC Options
# DPCPP_FLAGS += -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# Files
CFILES := 
CXXFILES := 
CUFILES := main.dp.cpp dwt.dp.cpp components.dp.cpp dwt_cuda/fdwt53.dp.cpp dwt_cuda/fdwt97.dp.cpp dwt_cuda/common.dp.cpp dwt_cuda/rdwt97.dp.cpp dwt_cuda/rdwt53.dp.cpp

# Includes
#INCLUDES := -I. -I$(CUDA_INSTALL_PATH)/include

include ../../common/make.config


# Common flags
COMMONFLAGS += $(INCLUDES) 
DPCPP_FLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS) -std=c99 
# DPCPP_FLAGS += -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

# Warning flags (from cuda common.mk)
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

CFLAGS += $(CWARN_FLAGS)
CXXFLAGS += $(CXXWARN_FLAGS)

# Debug/release flags
ifeq ($(dbg),1)
    COMMONFLAGS += -g 
    DPCPP_FLAGS   += -D_DEBUG
    CXXFLAGS    += -D_DEBUG
    CFLAGS      += -D_DEBUG

    ifeq ($(cudagdb),1)
        DPCPP_FLAGS += -G
    endif
else 
    DPCPP_FLAGS += -O2 
    #DPCPP_FLAGS   += --compiler-options -fno-strict-aliasing
    CXXFLAGS    += -fno-strict-aliasing
    CFLAGS      += -fno-strict-aliasing
endif

ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

#ifeq ($(performancetest),1)
#    COMMONFLAGS += -DGPU_DWT_TESTING
#endif

#ifdef OUTPUT
#	override OUTPUT = -DOUTPUT
#endif

OUTPUT = -DOUTPUT

# Compilers
# SYCL_C := dpcpp
CC := gcc
#SYCL_C := dpcpp -fPIC
# SYCL_C := $(SYCL_ROOT)/bin/clang++
# #NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc
# SYCL_C := $(SYCL_ROOT)/bin/clang++
# Generate object files list
COBJS=$(CFILES:.c=.c.o)
CXXOBJS=$(CXXFILES:.cpp=.cpp.o)
CUOBJS=$(CUFILES:.dp.cpp=.dp.cpp.o)

.SUFFIXES: .c.o .cpp.o .dp.cpp.o dp.cpp 

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.dp.cpp.o: %.dp.cpp
	$(SYCL_C) $(OUTPUT) $(DPCPP_FLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(SYCL_C) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(COBJS) $(CXXOBJS) $(CUOBJS) 
	$(SYCL_C) -o $(EXECUTABLE) $(COBJS) $(CXXOBJS) $(CUOBJS) $(DPCPP_FLAGS)

clean:
	rm -f $(COBJS) $(CXXOBJS) $(CUOBJS) $(EXECUTABLE)
	rm *.bmp.dwt.*		