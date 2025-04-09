#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
//--------------------------------------------------------------------------UTIL
// CODE---------------------------------------------------------------
#define TILE_WIDTH 16
int **create_matrix(int rows) {
  int **matrix = (int **)malloc(rows * sizeof(int *));
  for (int i = 0; i < rows; ++i) {
    matrix[i] = (int *)malloc(rows * (sizeof(int)));
  }
  return matrix;
}
void populate(int **matrix, int rows) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < rows; j++) {
      matrix[i][j] = rand() % 100;
    }
  }
}
void print_matrix(int **matrix, int rows) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < rows; j++) {
      printf("%d ", matrix[i][j]);
    }
    printf("\n");
  }
}
int *Linear(int **matrix, int rows) {
  int *l_m = (int *)malloc(rows * rows * sizeof(int));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < rows; j++) {
      int idx = i * rows + j;
      l_m[idx] = matrix[i][j];
    }
  }
  return l_m;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------kernel--------------------------------------------------------------
__global__ void matmulKernel(int *m1_d, int *m2_d, int *m3_d, int rows) {
  __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  int Pval = 0;
  for (int i = 0; i < rows / TILE_WIDTH; ++i) {

    Mds[ty][tx] = m1_d[row * rows + i * TILE_WIDTH + tx];
    Nds[ty][tx] = m2_d[(i * TILE_WIDTH + ty) * rows + col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pval += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  m3_d[row * rows + col] = Pval;
}

int *mat_mul(int **matrix_1, int **matrix_2, int rows) {
  int *prod_mtx = (int *)malloc(rows * rows * sizeof(int));
  // while working with cuda we have to linearize the matrix
  int *m1_h = Linear(matrix_1, rows), *m2_h = Linear(matrix_2, rows);
  int size = rows * rows * sizeof(int);

  int *m1_d, *m2_d, *m3_d;
  // Memory allocation
  cudaMalloc((void **)&m1_d, size);
  cudaMalloc((void **)&m2_d, size);
  cudaMalloc((void **)&m3_d, size);
  // Memory copy
  cudaMemcpy(m1_d, m1_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(m2_d, m2_h, size, cudaMemcpyHostToDevice);
  // kernel
  dim3 threadperblock(16, 16);
  dim3 blockspergrid((rows + threadperblock.x - 1) / threadperblock.x,
                     (rows + threadperblock.y - 1) / threadperblock.y);

  matmulKernel<<<blockspergrid, threadperblock>>>(m1_d, m2_d, m3_d, rows);
  cudaMemcpy(prod_mtx, m3_d, size, cudaMemcpyDeviceToHost);
  free(m1_h);
  free(m2_h);
  cudaFree(m1_d);
  cudaFree(m3_d);
  cudaFree(m2_d);
  return prod_mtx;
}

int main(int argc, char *argv[]) {
  int rows = 32;
  int **matrix_1 = create_matrix(rows);
  int **matrix_2 = create_matrix(rows);
  populate(matrix_1, rows);
  populate(matrix_2, rows);
  printf("THE FIRST MATRIX IS\n");
  print_matrix(matrix_1, rows);
  printf("THE SECOND MATRIX IS\n");
  print_matrix(matrix_2, rows);
  int *matrix_3 = mat_mul(matrix_1, matrix_2, rows);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < rows; j++) {
      printf("%d ", matrix_3[i * rows + j]);
    }
    printf("\n");
  }
  for (int i = 0; i < rows; ++i) {
    free(matrix_1[i]);
    free(matrix_2[i]);
  }
  free(matrix_1);
  free(matrix_2);
  free(matrix_3);
  return EXIT_SUCCESS;
}
