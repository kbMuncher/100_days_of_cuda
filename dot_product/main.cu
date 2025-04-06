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

// delinearize the matrix
void delinearize(matrix K, int *l) {

  for (int i = 0; i < K.rows; ++i) {
    for (int j = 0; j < K.cols; ++j) {
      K.mt[i][j] = l[i * K.cols + j];
    }
  }
}

__global__ void dotKerenl(int *A, int *B, int *C, int A_rows, int A_cols,
                          int B_cols) {
  int rows = threadIdx.y + blockIdx.y * blockDim.y;
  int cols = threadIdx.x + blockIdx.x * blockDim.x;
  if (rows < A_rows && cols < B_cols) {
    int sum = 0;
    for (int i = 0; i < A_cols; ++i) {
      sum += A[rows * A_cols + i] * B[i * B_cols + cols];
    }
    C[rows * B_cols + cols] = sum;
  }
}
void matrixMulKerenel(matrix A, matrix B, matrix C) {
  // get linear versions
  int *A_l = linearize(A), *B_l = linearize(B),
      *C_l = (int *)malloc(C.rows * C.cols * sizeof(int));
  ;
  int *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, A.rows * A.cols * sizeof(int));
  cudaMalloc((void **)&B_d, B.rows * B.cols * sizeof(int));
  cudaMalloc((void **)&C_d, C.rows * C.cols * sizeof(int));

  cudaMemcpy(A_d, A_l, A.rows * A.cols * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_l, B.rows * B.cols * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threadPerBlock(16, 16);
  dim3 blocksPerGrid((C.cols + threadPerBlock.x - 1) / threadPerBlock.x,
                     (C.rows + threadPerBlock.y - 1) / threadPerBlock.y);

  // kernel call
  dotKerenl<<<blocksPerGrid, threadPerBlock>>>(A_d, B_d, C_d, A.rows, A.cols,
                                               B.cols);
  cudaMemcpy(C_l, C_d, C.rows * C.cols * sizeof(int), cudaMemcpyDeviceToHost);
  free(A_l);
  free(B_l);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  delinearize(C, C_l);
  free(C_l);
}

int main() {
  matrix A;
  matrix B;

  A.rows = get_rows(1);
  A.cols = get_cols(1);
  B.rows = get_rows(2);
  B.cols = get_cols(2);

  //-----------------------BUILDING A MATRIX--------------------------
  A.mt = (int **)malloc(A.rows * sizeof(int *));
  for (int i = 0; i < A.rows; ++i) {
    A.mt[i] = (int *)malloc(A.cols * sizeof(int));
  }
  B.mt = (int **)malloc(B.rows * sizeof(int *));
  for (int i = 0; i < B.rows; ++i) {
    B.mt[i] = (int *)malloc(B.cols * sizeof(int));
  }

  // Populating the matrix
  populate_matrix(A);
  populate_matrix(B);

  if (A.cols != B.rows) {
    printf("No proper dimensions exiting...............\n");

    for (int i = 0; i < A.rows; ++i)
      free(A.mt[i]);
    free(A.mt);
    for (int i = 0; i < B.rows; ++i)
      free(B.mt[i]);
    free(B.mt);

    exit(0);
  } else {
    matrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.mt = (int **)malloc(C.rows * sizeof(int *));
    for (int i = 0; i < C.rows; ++i) {
      C.mt[i] = (int *)malloc(C.cols * sizeof(int));
    }
    printf("----------MATRIX A---------------\n");
    Print_matrix(A);
    printf("----------MATRIX B---------------\n");
    Print_matrix(B);
    printf(
        "--------------------MULTIPLYING THE MATRICES---------------------\n");
    matrixMulKerenel(A, B, C);

    Print_matrix(C);
    for (int i = 0; i < C.rows; ++i) {
      free(C.mt[i]);
    }
    free(C.mt);
  }
}
