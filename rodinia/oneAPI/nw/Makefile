CXX = dpcpp
TARGET = needleman_wunsch
SRCS = src/needle.dp.cpp
DEPS = src/needle_kernel.dp.cpp src/needle.h

# 'make time=1' compiles for time measurement
ifeq ($(time),1)
	TIMER = -DTIME_IT
endif
# Use predefined implicit rules and add one for *.cu files.
#%.o: %.cu
#	$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $< -o $@

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CXX) $(SRCS) $(TIMER) -o $@

run: $(TARGET)
	./$(TARGET) 4096 16

.PHONY: clean
clean:
	rm -f $(TARGET) *.o result.txt
