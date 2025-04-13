#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

typedef struct {
  int rows;
  int cols;
  int **mt;
} matrix;

void populate_matrix(matrix K) {
  for (int i = 0; i < K.rows; ++i) {
    for (int j = 0; j < K.cols; ++j) {
      K.mt[i][j] = rand() % 10;
    }
  }
}

int *linearize(matrix k) {
  int *L = (int *)malloc(k.rows * k.cols * sizeof(int));
  for (int i = 0; i < k.rows; i++) {
    for (int j = 0; j < k.cols; j++) {
      L[i * k.cols + j] = k.mt[i][j];
    }
  }
  return L;
}

// 1D convkernel [-1,0,1]
__global__ void conv1d_kernel(int *input, int *output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 1 && i < size - 1) {
    output[i] = input[i - 1] * 1 + input[i] * 0 + input[i + 1] * (-1);
  }
}

int main() {
  matrix A;
  printf("Enter rows and cols of matrix:\n");
  scanf("%d %d", &A.rows, &A.cols);

  A.mt = (int **)malloc(A.rows * sizeof(int *));
  for (int i = 0; i < A.rows; ++i) {
    A.mt[i] = (int *)malloc(A.cols * sizeof(int));
  }

  populate_matrix(A);
  int size = A.rows * A.cols;

  int *h_input = linearize(A);
  int *h_output = (int *)malloc(size * sizeof(int));
  int *d_input, *d_output;

  cudaMalloc(&d_input, size * sizeof(int));
  cudaMalloc(&d_output, size * sizeof(int));
  cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

  int blockSize = 256;
  int gridSize = (size + blockSize - 1) / blockSize;

  conv1d_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);

  cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

  printf("\nOriginal flattened matrix:\n");
  for (int i = 0; i < size; ++i) {
    printf("%d ", h_input[i]);
  }

  printf("\n\nAfter 1D convolution with filter [1 0 -1]:\n");
  for (int i = 0; i < size; ++i) {
    printf("%d ", h_output[i]);
  }

  for (int i = 0; i < A.rows; ++i)
    free(A.mt[i]);
  free(A.mt);
  free(h_input);
  free(h_output);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
