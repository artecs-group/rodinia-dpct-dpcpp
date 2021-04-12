#Configure
#CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_INSTALL_PATH = $(CUDA_DIR)
EXECUTABLE := dwt2d

# Debug options
dbg=0
cudagdb=0 #compile for use with cuda-gdb, note that 'dbg' must be 1 as well
performancetest=1

# NVCC Options
#NVCCFLAGS += -arch sm_20

# Files
CFILES := 
CXXFILES := 
CUFILES := main.dp.cpp dwt.dp.cpp components.dp.cpp dwt_cuda/fdwt53.dp.cpp dwt_cuda/fdwt97.dp.cpp dwt_cuda/common.dp.cpp dwt_cuda/rdwt97.dp.cpp dwt_cuda/rdwt53.dp.cpp

# Includes
#INCLUDES := -I. -I$(CUDA_INSTALL_PATH)/include

include ../../common/make.config


# Common flags
COMMONFLAGS += $(INCLUDES) 
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS) -std=c99 
LDFLAGS += -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

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
    NVCCFLAGS   += -D_DEBUG
    CXXFLAGS    += -D_DEBUG
    CFLAGS      += -D_DEBUG

    ifeq ($(cudagdb),1)
        NVCCFLAGS += -G
    endif
else 
    COMMONFLAGS += -O2 
    #NVCCFLAGS   += --compiler-options -fno-strict-aliasing
    CXXFLAGS    += -fno-strict-aliasing
    CFLAGS      += -fno-strict-aliasing
endif

#ifeq ($(performancetest),1)
#    COMMONFLAGS += -DGPU_DWT_TESTING
#endif

#ifdef OUTPUT
#	override OUTPUT = -DOUTPUT
#endif

OUTPUT = -DOUTPUT

# Compilers
CXX := dpcpp
CC := gcc
LINK := dpcpp -fPIC
#NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc
NVCC := dpcpp
# Generate object files list
COBJS=$(CFILES:.c=.c.o)
CXXOBJS=$(CXXFILES:.cpp=.cpp.o)
CUOBJS=$(CUFILES:.dp.cpp=.dp.cpp.o)

.SUFFIXES: .c.o .cpp.o .dp.cpp.o dp.cpp 

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.dp.cpp.o: %.dp.cpp
	$(NVCC) $(OUTPUT) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(COBJS) $(CXXOBJS) $(CUOBJS) 
	$(LINK) -o $(EXECUTABLE) $(COBJS) $(CXXOBJS) $(CUOBJS) $(LDFLAGS)

clean:
	rm -f $(COBJS) $(CXXOBJS) $(CUOBJS) $(EXECUTABLE)
	rm *.bmp.dwt.*		