#include <cstdio>
#include <opencv2/opencv.hpp>

__global__ void boxBlurKernel(unsigned char *input, unsigned char *output,
                              int width, int height) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col >= width || row >= height)
    return;

  int kernelSize = 3;
  int half = kernelSize / 2;
  int rSum = 0, gSum = 0, bSum = 0;
  int count = 0;

  for (int ky = -half; ky <= half; ky++) {
    for (int kx = -half; kx <= half; kx++) {
      int x = min(max(col + kx, 0), width - 1);
      int y = min(max(row + ky, 0), height - 1);
      int offset = (y * width + x) * 3;
      bSum += input[offset];
      gSum += input[offset + 1];
      rSum += input[offset + 2];
      count++;
    }
  }

  int idx = (row * width + col) * 3;
  output[idx] = bSum / count;
  output[idx + 1] = gSum / count;
  output[idx + 2] = rSum / count;
}
int main() {
  cv::Mat img = cv::imread("bl1.jpg");

  int width = img.cols;
  int height = img.rows;
  int imgSize = width * height * 3;  // colour format(BGR)
  int graySize = width * height * 3; // grayscale

  unsigned char *h_input = img.data;
  unsigned char *h_output = new unsigned char[graySize];

  unsigned char *d_input, *d_output;
  cudaMalloc((void **)&d_input, imgSize);
  cudaMalloc((void **)&d_output, graySize);

  cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);

  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  boxBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
  cudaMemcpy(h_output, d_output, graySize, cudaMemcpyDeviceToHost);

  cv::Mat output_img(height, width, CV_8UC3, h_output);

  cv::imshow("Grayscale Image", output_img);
  cv::imwrite("blur.jpg", output_img);
  cv::waitKey(0);

  cudaFree(d_input);
  cudaFree(d_output);
  delete[] h_output;

  return 0;
}
