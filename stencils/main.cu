#include <cmath>
#include <cstdio>
#include <math.h>
#include <stdio.h>

#define N 512          // grids
#define BLOCK_SIZE 256 // threads per block
#define PI 3.14159
#define H (2 * PI / N) // STEP SIZE (in book it was PI/6)

__global__ void derivate_1D(float *input, float *output, int n, float h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // i=0 and i=n-1 are boundary consditions since our stencil is in form
  // [i-h,i-i+h] so appropriate formulae applied
  if (i == 0) {
    output[i] = (input[i + 1] - input[i]) / h; // Forward diff
  } else if (i == n - 1) {
    output[i] = (input[i] - input[i - 1]) / h; // Backward diff
  } else {
    output[i] = (input[i + 1] - input[i - 1]) / (2 * h); // Central diff
  }
}

int main() {
  float *x = new float[N];
  float *sin_h = new float[N];
  float *sin_d, *sin_result, *sin_result_h = new float[N];

  cudaMalloc(&sin_d, N * sizeof(float));
  cudaMalloc(&sin_result, N * sizeof(float));

  // POPULATING THE !D GRID
  for (int i = 0; i < N; i++) {
    x[i] = i * H;          // FIgure 8.1B
    sin_h[i] = sinf(x[i]); // Figure 8.1C
  }
  cudaMemcpy(sin_d, sin_h, N * sizeof(float), cudaMemcpyHostToDevice);

  derivate_1D<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
      sin_d, sin_result, N, H);
  cudaMemcpy(sin_result_h, sin_result, N * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(sin_d);
  cudaFree(sin_result);

  printf("i\t|x\t|RESULT\t|ACTUAL(COS)\t|\tO(H^2)\n");
  for (int i = 0; i < 10; i++) {
    printf("%2d\t%.4f\t%.6f\t%.6f\t%.6f\n", i, x[i], sin_result_h[i],
           cosf(x[i]), cosf(x[i]) - sin_result_h[i]);
  }

  delete[] x;
  delete[] sin_h;
  delete[] sin_result_h;
  return 0;
}
