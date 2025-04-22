
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <vector>

using json = nlohmann::json;

__global__ void compute_gradients(const double *x, const double *y, double *dw,
                                  double *db, double w, double b, int N) {
  __shared__ double local_dw[256];
  __shared__ double local_db[256];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  double err = 0.0;
  if (idx < N) {
    double y_pred = w * x[idx] + b;
    err = y_pred - y[idx];
    local_dw[tid] = err * x[idx];
    local_db[tid] = err;
  } else {
    local_dw[tid] = 0.0;
    local_db[tid] = 0.0;
  }

  __syncthreads();

  // Reduce within block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      local_dw[tid] += local_dw[tid + s];
      local_db[tid] += local_db[tid + s];
    }
    __syncthreads();
  }

  // Add to global gradients (atomic)
  if (tid == 0) {
    atomicAdd(dw, local_dw[0]);
    atomicAdd(db, local_db[0]);
  }
}

void linear_regression_gpu(const double *h_x, const double *h_y, int N,
                           double &w, double &b, float &time_ms) {
  double *d_x, *d_y;
  double *d_dw, *d_db;
  cudaMalloc(&d_x, N * sizeof(double));
  cudaMalloc(&d_y, N * sizeof(double));
  cudaMalloc(&d_dw, sizeof(double));
  cudaMalloc(&d_db, sizeof(double));

  cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice);

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
    cudaMemset(d_dw, 0, sizeof(double));
    cudaMemset(d_db, 0, sizeof(double));

    compute_gradients<<<gridSize, blockSize>>>(d_x, d_y, d_dw, d_db, w, b, N);
    cudaDeviceSynchronize();

    double dw, db;
    cudaMemcpy(&dw, d_dw, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&db, d_db, sizeof(double), cudaMemcpyDeviceToHost);

    dw /= N;
    db /= N;

    w -= lr * dw;
    b -= lr * db;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_dw);
  cudaFree(d_db);
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

  results["cuda_optimised"] = {
      {"w", round(w * 10000) / 10000},
      {"b", round(b * 10000) / 10000},
      {"time", round(time_ms / 1000.0 * 10000) / 10000} // Convert ms to sec
  };

  std::ofstream out("results.json");
  out << std::setw(4) << results;
  out.close();

  std::cout << "[✓] CUDA Linear Regression done: w=" << w << ", b=" << b
            << ", time=" << time_ms / 1000.0 << "s\n";
  return 0;
}
