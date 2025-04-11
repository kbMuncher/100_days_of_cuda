#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct {
  int rows;
  int cols;
  int **mt;
} matrix;

int get_rows(int n) {
  printf("Enter rows of %d matrix\n", n);
  int r;
  scanf("%d", &r);
  return r;
}
int get_cols(int n) {
  printf("Enter cols of %d matrix\n", n);
  int r;
  scanf("%d", &r);
  return r;
}

void populate_matrix(matrix K) {
  for (int i = 0; i < K.rows; ++i) {
    for (int j = 0; j < K.cols; ++j) {
      K.mt[i][j] = rand() % 100;
    }
  }
}

void Print_matrix(matrix K) {
  for (int i = 0; i < K.rows; ++i) {
    for (int j = 0; j < K.cols; ++j) {
      printf("%d ", K.mt[i][j]);
    }
    printf("\n");
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

__global__ void bleh_r(int *A, int *B, int rows, int cols) {
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < rows && col < cols) {
    B[row * cols + col] = A[row * cols + col] * 2;
  }
}
__global__ void bleh_c(int *A, int *B, int rows, int cols) {
  int row = threadIdx.y + blockDim.y + blockIdx.y;
  int col = threadIdx.x + blockDim.x + blockIdx.x;

  if (row < rows && col < cols) {
    B[col * rows + row] = A[col * rows + row] * 2;
  }
}

void time_taken(matrix A, char *orientation) {
  int *m_h = linearize(A);
  int *m_d, *o_d;
  cudaMalloc(&m_d, A.rows * A.cols * sizeof(int));
  cudaMalloc(&o_d, A.rows * A.cols * sizeof(int));
  cudaMemcpy(m_d, m_h, A.rows * A.cols * sizeof(int), cudaMemcpyHostToDevice);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  // kerenl

  dim3 block(16, 16);
  dim3 grid((A.cols + block.x - 1) / block.x, (A.rows + block.y - 1) / block.y);

  if (orientation[0] == 'R' || orientation[0] == 'r') {
    bleh_r<<<grid, block>>>(m_d, o_d, A.rows, A.cols);

  } else {

    bleh_c<<<grid, block>>>(m_d, o_d, A.rows, A.cols);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("The time taken to read atrix %s is %f\n", orientation, ms);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(m_d);
  cudaFree(o_d);
  free(m_h);
}
int main() {
  matrix A;

  A.rows = get_rows(1);
  A.cols = get_cols(1);

  //-----------------------BUILDING A MATRIX--------------------------
  A.mt = (int **)malloc(A.rows * sizeof(int *));
  for (int i = 0; i < A.rows; ++i) {
    A.mt[i] = (int *)malloc(A.cols * sizeof(int));
  }

  // Populating the matrix
  char ty[4];
  populate_matrix(A);
  printf("Enter the orientation you would like to operate on (ROWS or COLS)\n");
  scanf("%s", ty);
  time_taken(A, ty);
  for (int i = 0; i < A.rows; ++i) {
    free(A.mt[i]);
  }
  free(A.mt);
}
