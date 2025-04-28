#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
using namespace std;
__global__ void histo_kernel(char *data, int length, int *histo) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < length) {
    int alpha_pos =
        data[i] - 'a'; // i.e range(97-122) - 97 putting all in range(0-25)
    if (alpha_pos >= 0 && alpha_pos < 26) {

      atomicAdd(&histo[alpha_pos / 4], 1);
    }
  }
}
int main() {
  string data;        // host data
  int count_histo[7]; // host data
  printf("ENTER TEXT\n");
  getline(cin, data);
  for (auto &ch : data) {
    ch = tolower(ch);
  }

  int size = data.length();
  // kerenl config
  char *data_d;
  int *histo;

  cudaMalloc(&data_d, size * sizeof(char));
  cudaMalloc(&histo, 26 * sizeof(int));
  cudaMemset(histo, 0, 7 * sizeof(int));

  cudaMemcpy(data_d, data.c_str(), size * sizeof(char), cudaMemcpyHostToDevice);

  // Configure kernel launch
  int block_size = 256; // Number of threads per block
  int grid_size = (size + block_size - 1) / block_size; // Number of blocks

  histo_kernel<<<grid_size, block_size>>>(data_d, size, histo);
  cudaDeviceSynchronize();
  cudaMemcpy(count_histo, histo, 7 * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(data_d);
  cudaFree(histo);

  printf("THE NUMBER OF OCCURENCE OF EACH ALPHABET IN THE SENTENCE IS\n");
  for (int i = 0; i < 7; ++i) {
    if (i < 6) {

      printf("%c %c %c %c %d\n", 'a' + 4 * i, 'b' + 4 * i, 'c' + 4 * i,
             'd' + 4 * i, count_histo[i]);
    } else {
      printf("%c %c  %d\n", 'a' + 4 * i, 'b' + 4 * i, count_histo[i]);
    }
  }
}
