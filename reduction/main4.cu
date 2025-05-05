// segmented sum reduction
#include <cstdlib>
#include <stdio.h>
#define SIZE 2048

__global__ void sum_reduction_kernel_segmented(int *input, int *output) {
  __shared__ int in_s[SIZE / 2];
  unsigned int segment = 2 * blockDim.x * blockIdx.x;
  unsigned int i = segment + threadIdx.x;
  unsigned int t = threadIdx.x; // ranges 0 2 4 6 8...........
  in_s[t] = input[i] + input[i + SIZE / 2];
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride >= 1;
       stride /= 2) { // ranges as 1 2 4 8......
    if (t < stride) {
      in_s[t] += in_s[t + stride];
    }
    __syncthreads();
  }
  if (t == 0) {
    atomicAdd(output, in_s[0]);
  }
}
int main() {
  int *arr = (int *)malloc(sizeof(int) * SIZE);

  for (int i = 0; i < SIZE; ++i) {
    arr[i] = 1;
  }
  // printf("THE ARRAY IS \n");
  // for (int i = 0; i < SIZE; ++i) {
  //   printf("%d ", arr[i]);
  // }
  // printf("\n");
  //
  int output_h, *output_d;
  int *arr_d;

  cudaMalloc(&arr_d, SIZE * sizeof(int));
  cudaMalloc(&output_d, sizeof(int));
  cudaMemset(&output_d, 0, sizeof(int));

  dim3 threads(SIZE / 2);

  cudaMemcpy(arr_d, arr, SIZE * sizeof(int), cudaMemcpyHostToDevice);

  sum_reduction_kernel_segmented<<<(SIZE + threads.x - 1) / threads.x,
                                   threads>>>(arr_d, output_d);
  cudaMemcpy(&output_h, output_d, sizeof(int), cudaMemcpyDeviceToHost);

  printf("THE SUM OF ENTIRE ARRAY IS %d\n", output_h);
  return 0;
}
