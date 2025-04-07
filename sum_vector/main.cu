#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int *A, int size, int *O) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size) {
    atomicAdd(O, A[i]);
  }
}

int vecAdd(int *A_h, int size) {
  int *A_d, *O_d;
  int O_h = 0;
  cudaMalloc((void **)&A_d, size * sizeof(int));
  cudaMalloc((void **)&O_d, sizeof(int));
  cudaMemcpy(A_d, A_h, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(O_d, &O_h, sizeof(int), cudaMemcpyHostToDevice);
  dim3 threadPerBlock(16);
  dim3 blocksPerGrid((size + threadPerBlock.x - 1) / threadPerBlock.x);
  kernel<<<blocksPerGrid, threadPerBlock>>>(A_d, size, O_d);
  cudaMemcpy(&O_h, O_d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(A_d);
  cudaFree(O_d);
  return O_h;
}

int main() {
  int arr[] = {1, 312, 41, 24, 124, 124, 124, 2};
  int size = sizeof(arr) / sizeof(int);
  int sumArr = 0;
  sumArr = vecAdd(arr, size);
  printf("The sum of Array is %d\n", sumArr);
}
