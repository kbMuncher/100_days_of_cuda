
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - (KERNEL_SIZE - 1))
__constant__ int F[KERNEL_SIZE * KERNEL_SIZE];

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
__global__ void conv2DKernel(int *m_h, int *output, int rows, int cols) {
  int row = threadIdx.y + OUT_TILE_DIM * blockIdx.y - ((KERNEL_SIZE - 1) / 2);
  int col = threadIdx.x + OUT_TILE_DIM * blockIdx.x - ((KERNEL_SIZE - 1) / 2);
  // loading input tiles
  __shared__ int N_s[IN_TILE_DIM][IN_TILE_DIM];
  // setting usuable elements to memory and ghost elements as zeros
  if (row >= 0 && row < rows && col >= 0 && col < cols) {
    N_s[threadIdx.y][threadIdx.x] = m_h[row * cols + col];
  } else
    N_s[threadIdx.y][threadIdx.x] = 0;
  __syncthreads(); // barrier sync
  int tileCOL = threadIdx.x - ((KERNEL_SIZE - 1) / 2);
  int tileROW = threadIdx.y - ((KERNEL_SIZE - 1) / 2);

  if (col >= 0 && col < cols && row >= 0 && row < rows) {
    if (tileCOL >= 0 && tileCOL < OUT_TILE_DIM && tileROW >= 0 &&
        tileROW < OUT_TILE_DIM) {
      int val = 0;
      for (int frow = 0; frow < KERNEL_SIZE; ++frow) {
        for (int fcol = 0; fcol < KERNEL_SIZE; ++fcol) {
          val += F[frow * KERNEL_SIZE + fcol] *
                 N_s[tileROW + frow][tileCOL + fcol];
        }
      }
      output[row * cols + col] = val;
    }
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
    int *m_h, *m_d, *o_h;

    o_h = (int *)malloc(A.rows * A.cols * sizeof(int));

    cudaMalloc(&m_h, A.rows * A.cols * sizeof(int));
    cudaMalloc(&m_d, A.rows * A.cols * sizeof(int));

    cudaMemcpy(m_h, A.mt, A.rows * A.cols * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, kerenl.mt, kerenl.rows * kerenl.cols * sizeof(int));
    dim3 block(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gird((A.cols + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
              (A.rows + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    conv2DKernel<<<gird, block>>>(m_h, m_d, A.rows, A.cols);
    cudaMemcpy(o_h, m_d, A.rows * A.cols * sizeof(int), cudaMemcpyDeviceToHost);
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
