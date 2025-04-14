#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

typedef struct {
  int rows;
  int cols;
  int *mt;
} matrix;

void populate_matrix(matrix K) {
  for (int i = 0; i < K.rows; ++i) {
    for (int j = 0; j < K.cols; ++j) {
      K.mt[i * K.cols + j] = rand() % 10;
    }
  }
}

void Print_matrix(matrix K) {
  for (int i = 0; i < K.rows; ++i) {
    for (int j = 0; j < K.cols; ++j) {
      printf("%d ", K.mt[i * K.cols + j]);
    }
    printf("\n");
  }
}
// kernel will alwasys be of size 3x3 for now
__global__ void conv2DKernel(int *m_h, int *k_h, int *output, int rows,
                             int cols) {
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  // setting boundaries since their is no padding
  if (col >= 1 && col < cols - 1 && row >= 1 && row < rows - 1) {
    int val = 0;
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        val += m_h[(row + y - 1) * cols + (col + x - 1)] * k_h[y * 3 + x];
      }
    }
    output[row * cols + col] = val;
  }
}
int main() {
  matrix A;
  printf("Enter rows of matrix:\n");
  scanf("%d", &A.rows);
  printf("Enter cols of matrix:\n");
  scanf("%d", &A.cols);
  A.mt = (int *)malloc(A.cols * A.rows * sizeof(int));

  populate_matrix(A);
  printf("THE ORIGINAL MATRIX IS\n");
  Print_matrix(A);
  char a;
  printf("Do you want to you a random kerenl?(y/n)\n");
  scanf(" %c", &a);
  if (a != 'y' && a != 'Y') {
    exit(0);
  } else {
    matrix kerenl;
    kerenl.cols = 3;
    kerenl.rows = 3;
    kerenl.mt = (int *)malloc(kerenl.cols * kerenl.rows * sizeof(int));
    populate_matrix(kerenl);
    Print_matrix(kerenl);
    printf("\nAPPLYING KERNEL.......................\n");
    int *m_h, *m_d, *k_h, *o_h;

    o_h = (int *)malloc(A.rows * A.cols * sizeof(int));

    cudaMalloc(&m_h, A.rows * A.cols * sizeof(int));
    cudaMalloc(&m_d, A.rows * A.cols * sizeof(int));
    cudaMalloc(&k_h, kerenl.cols * kerenl.rows * sizeof(int));
    cudaMemcpy(m_h, A.mt, A.rows * A.cols * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(k_h, kerenl.mt, kerenl.rows * kerenl.cols * sizeof(int),
               cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 gird((A.cols + block.x - 1) / block.x,
              (A.rows + block.y - 1) / block.y);
    conv2DKernel<<<gird, block>>>(m_h, k_h, m_d, A.rows, A.cols);
    cudaMemcpy(o_h, m_d, A.rows * A.cols * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(k_h);
    cudaFree(m_h);
    cudaFree(m_d);
    printf("THE RESULING MARIX IS\n");
    for (int i = 0; i < A.rows; ++i) {
      for (int j = 0; j < A.cols; ++j) {
        printf("%d ", o_h[i * A.cols + j]);
      }
      printf("\n");
    }

    free(A.mt);
    free(o_h);
    free(kerenl.mt);
  }
}
