CXX ?= g++
BUILD ?= build/m16
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude

.PHONY: m16-contract m16-clean

m16-contract:
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) tests/m16_disaggregated_serving_contract.cpp src/disaggregated_serving.cpp -o $(BUILD)/m16_contract
	$(BUILD)/m16_contract

m16-clean:
	rm -rf $(BUILD)
