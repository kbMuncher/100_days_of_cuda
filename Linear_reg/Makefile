# Makefile

.PHONY: all clean

PYTHON=python3
CUDA_DIR=cuda
BINARIES=$(CUDA_DIR)/main $(CUDA_DIR)/main2

all: gen lr cuda plot clean

gen:
	@$(PYTHON) util.py --gen

lr:
	@$(PYTHON) util.py --lr

cuda:
	@nvcc $(CUDA_DIR)/main.cu -o $(CUDA_DIR)/main
	@nvcc $(CUDA_DIR)/main2.cu -o $(CUDA_DIR)/main2
	@$(CUDA_DIR)/main
	@$(CUDA_DIR)/main2

plot:
	@$(PYTHON) util.py --plot

clean:
	@rm -f $(BINARIES)
	@rm -f x.bin
	@rm -f y.bin

