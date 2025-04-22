
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <vector>

using json = nlohmann::json;

__global__ void predict_and_error(const double *x, const double *y,
                                  double *errors, double *gradients_w,
                                  double *gradients_b, double w, double b,
                                  int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    double y_pred = w * x[idx] + b;
    double error = y_pred - y[idx];
    errors[idx] = error;
    gradients_w[idx] = error * x[idx];
    gradients_b[idx] = error;
  }
}

void linear_regression_gpu(const double *h_x, const double *h_y, int N,
                           double &w, double &b, float &time_ms) {
  double *d_x, *d_y, *d_errors, *d_grad_w, *d_grad_b;
  cudaMalloc(&d_x, N * sizeof(double));
  cudaMalloc(&d_y, N * sizeof(double));
  cudaMalloc(&d_errors, N * sizeof(double));
  cudaMalloc(&d_grad_w, N * sizeof(double));
  cudaMalloc(&d_grad_b, N * sizeof(double));

  cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice);

  std::vector<double> h_grad_w(N), h_grad_b(N);

  w = 0.0;
  b = 0.0;
  double lr = 0.01;
  int epochs = 1000;
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    predict_and_error<<<gridSize, blockSize>>>(d_x, d_y, d_errors, d_grad_w,
                                               d_grad_b, w, b, N);
    cudaMemcpy(h_grad_w.data(), d_grad_w, N * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_b.data(), d_grad_b, N * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum_dw = 0.0, sum_db = 0.0;
    for (int i = 0; i < N; ++i) {
      sum_dw += h_grad_w[i];
      sum_db += h_grad_b[i];
    }

    w -= lr * (sum_dw / N);
    b -= lr * (sum_db / N);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_errors);
  cudaFree(d_grad_w);
  cudaFree(d_grad_b);
}

int main() {
  std::ifstream x_file("x.bin", std::ios::binary | std::ios::ate);
  std::ifstream y_file("y.bin", std::ios::binary | std::ios::ate);
  size_t size = x_file.tellg();
  int N = size / sizeof(double);
  x_file.seekg(0);
  y_file.seekg(0);

  std::vector<double> h_x(N), h_y(N);
  x_file.read(reinterpret_cast<char *>(h_x.data()), size);
  y_file.read(reinterpret_cast<char *>(h_y.data()), size);
  x_file.close();
  y_file.close();

  double w, b;
  float time_ms;
  linear_regression_gpu(h_x.data(), h_y.data(), N, w, b, time_ms);

  // Load and update results.json
  std::ifstream in("results.json");
  json results;
  if (in)
    in >> results;
  in.close();

  results["cuda_unoptimised"] = {
      {"w", round(w * 10000) / 10000},
      {"b", round(b * 10000) / 10000},
      {"time", round(time_ms / 1000.0 * 10000) / 10000}};

  std::ofstream out("results.json");
  out << std::setw(4) << results;
  out.close();

  std::cout << "[✓] Plain CUDA Linear Regression done:\n";
  std::cout << "    w = " << w << "\n";
  std::cout << "    b = " << b << "\n";
  std::cout << "    time = " << time_ms / 1000.0 << " s\n";
  return 0;
}
