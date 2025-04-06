#include <cstdio>
#include <opencv2/opencv.hpp>

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
__global__ void sobelEdgeKernel(unsigned char *Pin, unsigned char *Pout,
                                int width, int height) {
  int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (col < width && row < height) {

    float gx = 0, gy = 0;

    for (int ky = -1; ky <= 1; ky++) {
      for (int kx = -1; kx <= 1; kx++) {
        int x = min(max(col + kx, 0), width - 1);
        int y = min(max(row + ky, 0), height - 1);
        int offset = y * width + x;

        float gray = Pin[offset];
        gx += gray * sobelX[ky + 1][kx + 1];
        gy += gray * sobelY[ky + 1][kx + 1];
      }
    }

    int idx = row * width + col;
    unsigned char edge = min(max((int)sqrtf(gx * gx + gy * gy), 0), 255);
    Pout[idx] = edge;
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
  unsigned char *h_edges = new unsigned char[graySize];

  unsigned char *d_input, *d_output, *d_edges;
  cudaMalloc((void **)&d_input, imgSize);
  cudaMalloc((void **)&d_output, graySize);
  cudaMalloc((void **)&d_edges, graySize);

  cudaMemcpy(d_input, h_input, imgSize, cudaMemcpyHostToDevice);
  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  grayKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

  sobelEdgeKernel<<<gridSize, blockSize>>>(d_output, d_edges, width, height);
  cudaMemcpy(h_edges, d_edges, graySize, cudaMemcpyDeviceToHost);

  cv::Mat output_img(height, width, CV_8UC1, h_edges);
  cv::imshow("Sobel Edge Detection", output_img);
  cv::imwrite("sobel.jpg", output_img);
  cv::waitKey(0);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_edges);
  delete[] h_output;
  delete[] h_edges;
  return 0;
}
