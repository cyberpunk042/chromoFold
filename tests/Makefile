CXX ?= g++
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Werror
SANITIZERS ?= -fsanitize=address,undefined -fno-omit-frame-pointer
ROOT := ..
BUILD := $(ROOT)/build
TARGET := $(BUILD)/kv_runtime_test

.PHONY: all run clean

all: $(TARGET)

$(TARGET): kv_runtime_test.cpp $(ROOT)/src/runtime/kv_runtime.cpp $(ROOT)/include/chromofold/kv_runtime.h $(ROOT)/include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(SANITIZERS) -I$(ROOT)/include kv_runtime_test.cpp $(ROOT)/src/runtime/kv_runtime.cpp -o $@

run: $(TARGET)
	ASAN_OPTIONS=detect_leaks=1 $(TARGET)

clean:
	rm -f $(TARGET)
