include ../../common/make.config

# SYCL_C = $(SYCL_ROOT)/bin/clang++
TARGET = needleman_wunsch
SRCS = src/needle.dp.cpp
DEPS = src/needle_kernel.dp.cpp src/needle.h
DPCPP_FLAGS = -g $(DPCPP_FLAGS)

ifeq ($(time),1)
	DPCPP_FLAGS += -DTIME_IT
endif

# Use predefined implicit rules and add one for *.cu files.
%.o: %.cu
	$(SYCL_C) $(DPCPP_FLAGS) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(SYCL_C) $(DPCPP_FLAGS) $(SRCS) -o $@

run: $(TARGET)
	./$(TARGET) 4096 16

.PHONY: clean
clean:
	rm -f $(TARGET) *.o result.txt
