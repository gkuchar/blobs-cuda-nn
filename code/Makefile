NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 --expt-extended-lambda
INCLUDES = -Iinclude

BIN_DIR = bin
SRC_DIR = src

TARGETS = $(BIN_DIR)/gen_blobs3 $(BIN_DIR)/main

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/gen_blobs3: $(BIN_DIR) $(SRC_DIR)/gen_blobs3.cu include/data_io.h
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(SRC_DIR)/gen_blobs3.cu -o $@

$(BIN_DIR)/main: $(BIN_DIR) $(SRC_DIR)/main.cu include/data_io.h include/thrust_nn.h
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(SRC_DIR)/main.cu -o $@

clean:
	rm -rf $(BIN_DIR)
