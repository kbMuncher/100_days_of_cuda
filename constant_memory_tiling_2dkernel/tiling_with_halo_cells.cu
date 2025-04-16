#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3
#define FILTER_RADIUS (KERNEL_SIZE - 1) / 2
#define TILE_DIM 32
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
__global__ void conv2DKernel(int *N, int *P, int width, int height) {

  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  int row = blockIdx.y * TILE_DIM + threadIdx.y;

  // Loading input tile
  __shared__ float N_s[TILE_DIM][TILE_DIM];
  if (row < height && col < width) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }

  __syncthreads();

  // Calculating output elements
  // Turning off threads at the edges of the block
  if (col < width && row < height) {
    int Pvalue = 0;

    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {

        if ((threadIdx.x + fCol) >= FILTER_RADIUS &&
            threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
            threadIdx.y + fRow >= FILTER_RADIUS &&
            threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM) {

          Pvalue += F[fRow * KERNEL_SIZE + fCol] *
                    N_s[threadIdx.y - FILTER_RADIUS + fRow]
                       [threadIdx.x - FILTER_RADIUS + fCol];

        } else {
          if ((row + fRow) >= FILTER_RADIUS &&
              (row - FILTER_RADIUS + fRow) < height &&
              (col + fCol) >= FILTER_RADIUS &&
              (col - FILTER_RADIUS + fCol) < width) {

            Pvalue += F[fRow * KERNEL_SIZE + fCol] *
                      N[(row - FILTER_RADIUS + fRow) * width +
                        (col - FILTER_RADIUS + fCol)];
          }
        }
      }
    }

    P[row * width + col] = Pvalue;
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
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 gird((A.cols + TILE_DIM - 1) / TILE_DIM,
              (A.rows + TILE_DIM - 1) / TILE_DIM);
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
