#include <cstdlib>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
double time_taken;
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}
void vec_add(float *A_h, float *B_h, float *C_h, int n) {
  int size = n * sizeof(float);
  float *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);
  // Memory copy
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
  // cals here man
  clock_t start, end;
  start = clock();
  vecAddKernel<<<ceil(n / 256.0), 257>>>(A_d, B_d, C_d, size);
  end = clock();
  time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  cudaMemcpy(C_d, C_h, size, cudaMemcpyDeviceToHost);
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main(int argc, char *argv[]) {
  float *A, *B, *C;
  A = (float *)malloc(sizeof(float) * 1000000);
  B = (float *)malloc(sizeof(float) * 1000000);
  C = (float *)malloc(sizeof(float) * 1000000);

  for (int i = 0; i < 1000000; ++i) {
    A[i] = i + 1;
    B[i] = i + 1000001;
  }
  vec_add(A, B, C, 1000000);
  printf("Time taken by GPU:- %f\n", time_taken);
  free(A);
  free(B);
  free(C);
  return EXIT_SUCCESS;
}
