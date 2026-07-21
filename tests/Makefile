CXX ?= g++
CXXFLAGS ?= -O2 -g -std=c++20 -Wall -Wextra -Wpedantic -Werror
SANITIZERS ?= -fsanitize=address,undefined -fno-omit-frame-pointer
ROOT := ..
BUILD := $(ROOT)/build
RUNTIME_TARGET := $(BUILD)/kv_runtime_test
REFERENCE_TARGET := $(BUILD)/kv_reference_test
TARGETS := $(RUNTIME_TARGET) $(REFERENCE_TARGET)

.PHONY: all run clean

all: $(TARGETS)

$(RUNTIME_TARGET): kv_runtime_test.cpp $(ROOT)/src/runtime/kv_runtime.cpp $(ROOT)/include/chromofold/kv_runtime.h $(ROOT)/include/chromofold/chromofold.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(SANITIZERS) -I$(ROOT)/include kv_runtime_test.cpp $(ROOT)/src/runtime/kv_runtime.cpp -o $@

$(REFERENCE_TARGET): kv_reference_test.cpp $(ROOT)/src/runtime/kv_reference.cpp $(ROOT)/include/chromofold/kv_reference.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(SANITIZERS) -I$(ROOT)/include kv_reference_test.cpp $(ROOT)/src/runtime/kv_reference.cpp -o $@

run: $(TARGETS)
	ASAN_OPTIONS=detect_leaks=1 $(RUNTIME_TARGET)
	ASAN_OPTIONS=detect_leaks=1 $(REFERENCE_TARGET)

clean:
	rm -f $(TARGETS)
