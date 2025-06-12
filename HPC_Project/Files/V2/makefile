# Makefile for V2 CUDA Neural Network

NVCC = nvcc
CFLAGS = -O2
LDFLAGS = -lm

EXE = nn_v2
SRC = nn.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(NVCC) $(CFLAGS) -o $(EXE) $(SRC)

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
