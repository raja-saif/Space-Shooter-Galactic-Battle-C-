# Makefile for V4 CUDA Neural Network (Tensor Core Attempt)

NVCC = nvcc

# --- IMPORTANT: Set this architecture flag to match your GPU ---
# sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere GA100), sm_86 (Ampere GA10x), etc.
# Check your GPU's Compute Capability!
CUDA_ARCH = -arch=sm_70

# Optimization level, Architecture flag, Host Compiler Warnings, Debug/Profile Info
CFLAGS = -O3 $(CUDA_ARCH) -Xcompiler -Wall -lineinfo

# Libraries needed: cuBLAS and Math library
LIBS = -lcublas -lm

# Target Executable Name
EXE = nn_v4
# Source file (ensure it matches your saved filename)
SRC = nn.cu

# Default target: build the executable
all: $(EXE)

# Rule to build the executable
$(EXE): $(SRC)
	$(NVCC) $(CFLAGS) -o $(EXE) $(SRC) $(LIBS)

# Rule to run the executable
run: $(EXE)
	./$(EXE)

# Rule to clean the executable
clean:
	rm -f $(EXE)

.PHONY: all run clean