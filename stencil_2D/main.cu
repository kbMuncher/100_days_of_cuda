#include <cmath>
#include <stdio.h>
#define N 32
#define BLOCK_SIZE 16 // 2D grid so 256
#define PI 3.14159f
#define H (2 * PI / N) // step size
__global__ void p_2D_derivative(float *input, float *dfdx, float *dfdy,
                                int width, int height, float h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    int idx = y * width + x;
    dfdx[idx] =
        (input[y * width + (x + 1)] - input[y * width + (x - 1)]) / (2 * h);
    dfdy[idx] =
        (input[(y + 1) * width + x] - input[(y - 1) * width + x]) / (2 * h);
  }
}
int main() {
  // EQUATION WILL BE f(x,y) = Sinx + Cosy
  float *f_x = new float[N * N];
  float *df_dx = new float[N * N];
  float *df_dy = new float[N * N];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float x = i * H;
      float y = j * H;
      f_x[j * N + i] = sinf(x) + cosf(y);
    }
  }

  float *fx_d, *df_dy_d, *df_dx_d;
  cudaMalloc(&fx_d, N * N * sizeof(float));
  cudaMalloc(&df_dx_d, N * N * sizeof(float));
  cudaMalloc(&df_dy_d, N * N * sizeof(float));

  cudaMemcpy(fx_d, f_x, N * N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  p_2D_derivative<<<grid, block>>>(fx_d, df_dx_d, df_dy_d, N, N, H);
  cudaDeviceSynchronize();
  cudaMemcpy(df_dx, df_dx_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(df_dy, df_dy_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(fx_d);
  cudaFree(df_dx_d);
  cudaFree(df_dy_d);
  printf("| x\t y |  df/dx |  COS(x) |  error |  df/dy |  -SIN(x) |  error \n");
  for (int j = 1; j < 5; j++) {
    for (int i = 1; i < 5; i++) {
      float x = i * H;
      float y = j * H;
      int idx = j * N + i;
      printf("| %.2f %.2f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f\n", x, y,
             df_dx[idx], cosf(x), fabsf(df_dx[idx] - cosf(x)), df_dy[idx],
             -sinf(y), fabsf(df_dy[idx] + sinf(y)));
    }
  }
  delete[] f_x;
  delete[] df_dy;
  delete[] df_dx;
}
