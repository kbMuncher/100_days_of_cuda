#include "opencv2/opencv.hpp"
#include <cstdio>

__global__ void grayKernel(unsigned char *Pin, unsigned char *Pout, int width,
                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int grayoffSet = row * width + col;
    int rgboffSet = grayoffSet * 3;
    unsigned char B = Pin[rgboffSet];
    unsigned char G = Pin[rgboffSet + 1];
    unsigned char R = Pin[rgboffSet + 2];

    // formula
    unsigned char L = (0.21f * R + 0.72f * G + 0.07f * B);

    Pout[grayoffSet] = L;
  }
}

int main() {
  cv::Mat img = cv::imread("bl1.jpg");

  int width = img.cols;
  int height = img.rows;
  int imgSize = width * height * 3; // colour format(BGR)
  int graySize = width * height;    // grayscale

  unsigned char *h_input = img.data;
  unsigned char *h_output = new unsigned char[graySize];

  unsigned char *d_input, *d_output;
  cudaMalloc((void **)&d_input, imgSize);
  cudaMalloc((void **)&d_output, graySize);

  cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);

  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  grayKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
  cudaMemcpy(h_output, d_output, graySize, cudaMemcpyDeviceToHost);

  cv::Mat output_img(height, width, CV_8UC1, h_output);

  cv::imshow("Grayscale Image", output_img);
  cv::imwrite("gray.jpg", output_img);
  cv::waitKey(0);

  cudaFree(d_input);
  cudaFree(d_output);
  delete[] h_output;

  return 0;
}
