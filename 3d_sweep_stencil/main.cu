#include <cuda_runtime.h>
#include <iostream>

#define N 8
#define BLOCK_SIZE 4

__constant__ float c0 = 0.01f;
__constant__ float c1 = 0.05f;
__constant__ float c2 = 0.09f;
__constant__ float c3 = 0.01f;
__constant__ float c4 = 0.09f;
__constant__ float c5 = 0.05f;
__constant__ float c6 = 0.01f;

__global__ void stencil_sweep_3D(float *in, float *out, int n) {
  int i = blockDim.z * blockIdx.z + threadIdx.z; // [N, N , _]
  int j = blockIdx.y * blockDim.y + threadIdx.y; //[N ,_, N]
  int k = blockIdx.x * blockDim.x + threadIdx.x; //[_,N,N]

  if (i > 0 && i < n - 1 && j > 0 && j < n - 1 && k > 0 && k < n - 1) {
    out[i * n * n + j * n + k] = c0 * in[i * n * n + j * n + k] +
                                 c1 * in[(i - 1) * n * n + j * n + k] +
                                 c2 * in[(i + 1) * n * n + j * n + k] +
                                 c3 * in[i * n * n + (j - 1) * n + k] +
                                 c4 * in[i * n * n + (j + 1) * n + k] +
                                 c5 * in[i * n * n + j * n + (k - 1)] +
                                 c6 * in[i * n * n + j * n + (k + 1)];
  }
}

int main() {
  unsigned int size = N * N * N * sizeof(float);
  float *in_h = new float[N * N * N];
  float *out_h = new float[N * N * N];

  // populate the matrix
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        in_h[i * N * N + j * N + k] = static_cast<float>(i * N * N + j * N + k);
      }
    }
  }
  // kernel config
  float *in_d, *out_d;
  cudaMalloc(&in_d, size);
  cudaMalloc(&out_d, size);

  cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(2, 2,
            2); // use (BLOCK_SIZE+N-1)/BLOCK_SIZE
                // now this means tioal threads per block is 4*4*4 = 64 amd
                // blocks per grid is 8 i.e thread per kernel launch is 512

  stencil_sweep_3D<<<grid, block>>>(in_d, out_d, N);

  cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);
  cudaFree(in_d);
  cudaFree(out_d);

  printf("____________________________________________________\n");
  printf("FIRST A SLICE FROM ORIGIANL MATRIX (lets take FIRST slcie\n");
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
      printf("%.2f ", in_h[1 * N * N + j * N + k]);
    }
    printf("\n");
  }
  printf("____________________________________________________\n");
  printf("NOW CORRESPONDING SLICE FROM THE OUTPUT \n");
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
      printf("%.2f ", out_h[1 * N * N + j * N + k]);
    }
    printf("\n");
  }
  delete[] in_h;
  delete[] out_h;
  return 0;
}
