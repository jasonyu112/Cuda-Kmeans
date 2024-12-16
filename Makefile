CC = g++
NVCC = nvcc
SRCS = ./src/*.cpp
INC = ./src/
CUDA_PATH ?= /usr/local/cuda-10.1
OPTS = -std=c++17 -Wall -Werror -O3 -I$(SRC_DIR) -I$(CUDA_PATH)/include
NVCCFLAGS = -O3 -arch=sm_75 -std=c++14 --expt-extended-lambda -I$(SRC_DIR) -I$(CUDA_PATH)/include


# Paths and files
SRC_DIR = ./src
BIN_DIR = ./bin
EXEC = $(BIN_DIR)/kmeans
LIBS = -L$(CUDA_PATH)/lib64 -lcudart

# Source and Object files
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CPP_OBJS = $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)
CU_OBJS = $(CU_SRCS:$(SRC_DIR)/%.cu=$(BIN_DIR)/%.o)

all: $(EXEC)

$(EXEC): $(CPP_OBJS) $(CU_OBJS) | $(BIN_DIR)
	$(CC) $(CPP_OBJS) $(CU_OBJS) -lpthread $(LIBS) -o $@

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CC) $(OPTS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

valgrind: $(EXEC)
	valgrind --leak-check=full --track-origins=yes $(EXEC) $(ARGS)

cuda-memcheck: $(EXEC)
	cuda-memcheck $(EXEC) $(ARGS)

nvprof: $(EXEC)
	nvprof $(EXEC) $(ARGS)