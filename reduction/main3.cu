// controlling memory divergence
#include <cstdlib>
#include <stdio.h>
#define SIZE 8

__global__ void sum_reduction_kernel(int *input, int *output) {
  __shared__ int in_s[SIZE / 2];
  unsigned int i = threadIdx.x; // ranges 0 2 4 6 8...........
  in_s[i] = input[i] + input[i + SIZE / 2];
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride >= 1;
       stride /= 2) { // ranges as 1 2 4 8......
    if (threadIdx.x < stride) {
      in_s[i] += in_s[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = in_s[0];
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
