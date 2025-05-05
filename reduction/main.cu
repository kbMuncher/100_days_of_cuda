// works only for SIZE that is a power of 2 and the threads must be of SIZE/2
// and block 1
#include <cstdlib>
#include <stdio.h>
#define SIZE 8

__global__ void sum_reduction_kernel(int *input, int *output) {
  unsigned int i = 2 * threadIdx.x; // ranges 0 2 4 6 8...........
  for (unsigned int stride = 1; stride <= blockDim.x;
       stride *= 2) { // ranges as 1 2 4 8......
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}
int main() {
  int *arr = (int *)malloc(sizeof(int) * SIZE);

  for (int i = 0; i < SIZE; ++i) {
    arr[i] = rand() % 10;
  }
  printf("THE ARRAY IS \n");
  for (int i = 0; i < SIZE; ++i) {
    printf("%d ", arr[i]);
  }
  printf("\n");

  int output_h, *output_d;
  int *arr_d;

  cudaMalloc(&arr_d, SIZE * sizeof(int));
  cudaMalloc(&output_d, sizeof(int));
  cudaMemset(&output_d, 0, sizeof(int));

  dim3 threads(SIZE / 2);

  cudaMemcpy(arr_d, arr, SIZE * sizeof(int), cudaMemcpyHostToDevice);

  sum_reduction_kernel<<<1, threads>>>(arr_d, output_d);
  cudaMemcpy(&output_h, output_d, sizeof(int), cudaMemcpyDeviceToHost);

  printf("THE SUM OF ENTIRE ARRAY IS %d\n", output_h);
  return 0;
}
