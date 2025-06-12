# Makefile for V3 CUDA Neural Network

NVCC = nvcc
CUDA_ARCH = -arch=sm_70
CFLAGS = -O3 $(CUDA_ARCH) -Xcompiler -Wall,-fopenmp -lineinfo
LIBS = -lcublas

EXE = nn_gpu_opt
SRC = nn.cu

all: $(EXE)

$(EXE): $(SRC)
	$(NVCC) $(CFLAGS) $(LIBS) -o $(EXE) $(SRC)

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)