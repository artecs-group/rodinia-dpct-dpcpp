include ../../common/make.config

CXX = $(SYCL_ROOT)/bin/clang++
TARGET = needleman_wunsch
SRCS = src/needle.dp.cpp
DEPS = src/needle_kernel.dp.cpp src/needle.h
CXX_FLAGS = -std=c++17 -g -I $(SYCL_INCLUDE_DIR) -I $(DPCT_INCLUDE_DIR) -L $(SYCL_LIB) $(SYCL_FLAGS)

ifeq ($(time),1)
	CXX_FLAGS += -DTIME_IT
endif

# Use predefined implicit rules and add one for *.cu files.
%.o: %.cu
	$(CXX) $(CXX_FLAGS) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CXX) $(CXX_FLAGS) $(SRCS) -o $@

run: $(TARGET)
	./$(TARGET) 4096 16

.PHONY: clean
clean:
	rm -f $(TARGET) *.o result.txt
