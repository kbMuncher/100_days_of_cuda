#include <cstdio>
#include <ctime>
#include <opencv2/opencv.hpp>

int main() {
  cv::Mat img = cv::imread("bl1.jpg");

  int width = img.cols;
  int height = img.rows;
  std::cout << "Image size: " << img.cols << " x " << img.rows << std::endl;
  std::cout << "Channels: " << img.channels() << std::endl;
  int imgSize = width * height * 3; // colour format(BGR)
  int graySize = width * height;    // grayscale

  unsigned char *h_input = img.data;
  unsigned char *h_output = new unsigned char[graySize];
  clock_t start, end;
  start = clock();
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int grayoffSet = i * width + j;
      int rgboffSet = grayoffSet * 3;
      unsigned char B = h_input[rgboffSet];
      unsigned char G = h_input[rgboffSet + 1];
      unsigned char R = h_input[rgboffSet + 2];

      // formula
      unsigned char L = (0.21f * R + 0.72f * G + 0.07f * B);

      h_output[grayoffSet] = L;
    }
  }
  end = clock();
  printf("time taken by CPU for conversion is %f\n",
         (float)(end - start) / CLOCKS_PER_SEC);
  delete[] h_output;
  return 0;
}
