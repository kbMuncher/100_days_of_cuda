
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 50
#define EPOCHS 1000
#define LEARNING_RATE 0.01

__global__ void compute_gradients(float *x, float *y, float m, float c,
                                  float *m_grad, float *c_grad) {
  int i = threadIdx.x;
  if (i < N) {
    float y_pred = m * x[i] + c;
    float error = y_pred - y[i];

    atomicAdd(m_grad, error * x[i] * 2.0f / N);
    atomicAdd(c_grad, error * 2.0f / N);
  }
}

int main() {
  float x[N], y[N];
  float m = 0.0f, c = 0.0f;
  float true_m = 2.5f, true_c = 1.0f;

  srand(0);
  for (int i = 0; i < N; i++) {
    x[i] = i * 10.0f / (N - 1);                       // linspace
    float noise = ((float)rand() / RAND_MAX) * 2 - 1; // random noise [-1, 1]
    y[i] = true_m * x[i] + true_c + noise;
  }

  // Device arrays
  float *d_x, *d_y;
  cudaMalloc((void **)&d_x, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    float m_grad = 0.0f, c_grad = 0.0f;
    float *d_m_grad, *d_c_grad;

    cudaMalloc((void **)&d_m_grad, sizeof(float));
    cudaMalloc((void **)&d_c_grad, sizeof(float));
    cudaMemcpy(d_m_grad, &m_grad, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_grad, &c_grad, sizeof(float), cudaMemcpyHostToDevice);

    compute_gradients<<<1, N>>>(d_x, d_y, m, c, d_m_grad, d_c_grad);

    cudaMemcpy(&m_grad, d_m_grad, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c_grad, d_c_grad, sizeof(float), cudaMemcpyDeviceToHost);

    m -= LEARNING_RATE * m_grad;
    c -= LEARNING_RATE * c_grad;

    cudaFree(d_m_grad);
    cudaFree(d_c_grad);
  }

  cudaFree(d_x);
  cudaFree(d_y);

  printf("Learned slope (m): %.3f\n", m);
  printf("Learned intercept (c): %.3f\n", c);

  return 0;
}
