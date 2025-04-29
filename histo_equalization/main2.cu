// A Histo Equalisation kernel on GrayScale image
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cstdlib>
#include <opencv2/opencv.hpp>

__global__ void histo_eq(unsigned char *in, int *histo, int width, int height) {
  int idx =
      blockDim.x * blockIdx.x +
      threadIdx.x; // accessing in linear form like 1D all elements in a line
  if (idx < width * height) {
    atomicAdd(&histo[in[idx]], 1);
  }
}

__global__ void cdf(unsigned char *in, unsigned char *out, unsigned char *cdf,
                    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = cdf[in[idx]];
  }
}
int main() {
  cv::Mat img = cv::imread("gray.jpg", cv::IMREAD_GRAYSCALE);
  int width = img.cols;
  int height = img.rows;
  int SIZE = width * height;
  unsigned char *h_input = img.data;
  unsigned char *h_output = new unsigned char[SIZE];
  int h_histo[256] = {0};
  unsigned char h_cdf[256] = {0};

  unsigned char *d_input, *d_output, *d_cdf;
  int *d_histo;
  cudaMalloc(&d_input, SIZE);
  cudaMalloc(&d_output, SIZE);
  cudaMalloc(&d_histo, 256 * sizeof(int));
  cudaMemcpy(d_input, h_input, SIZE, cudaMemcpyHostToDevice);
  cudaMemset(d_histo, 0, sizeof(int) * 256);
  dim3 threads(256);
  dim3 blocks((SIZE + 255) / 256);
  histo_eq<<<blocks, threads>>>(d_input, d_histo, width, height);
  cudaDeviceSynchronize();
  cudaMemcpy(h_histo, d_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

  int cdf_sum = 0;
  for (int i = 0; i < 256; ++i) {
    cdf_sum += h_histo[i];
    h_cdf[i] = static_cast<unsigned char>((cdf_sum * 255) / SIZE);
  }
  cudaMalloc(&d_cdf, 256);
  cudaMemcpy(d_cdf, h_cdf, 256, cudaMemcpyHostToDevice);
  cdf<<<blocks, threads>>>(d_input, d_output, d_cdf, SIZE);
  cudaDeviceSynchronize();
  cudaMemcpy(h_output, d_output, SIZE, cudaMemcpyDeviceToHost);

  cv::Mat output_img(height, width, CV_8UC1, h_output);
  cv::imwrite("HISTO_EQ.jpg", output_img);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_cdf);
  cudaFree(d_histo);
  delete[] h_output;

  return 0;
}
