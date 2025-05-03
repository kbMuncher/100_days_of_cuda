// Interleaved Partioning
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;

#define NUM_BINS 7 // 26 /4
__global__ void histo_private_kernel(char *data, unsigned int length,
                                     unsigned int *histo) {
  // private bins
  __shared__ unsigned int histo_s[NUM_BINS];
  for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    histo_s[bin] = 0u;
  }

  __syncthreads();
  // create Histogram
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned int i = tid * 4; i < length && i < (tid + 1) * 4; i++) {
    int alpha_pos = data[i] - 'a';
    if (alpha_pos >= 0 & alpha_pos < 26) {
      atomicAdd(&histo_s[alpha_pos / 4], 1);
    }
  }
  __syncthreads();
  // Commiting to Global memory
  for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned int binValue = histo_s[bin];
    if (binValue > 0) {
      atomicAdd(&histo[bin], binValue);
    }
  }
}

int main() {
  string data;
  unsigned int count_histo[NUM_BINS];
  printf("ENTER TEXT\n");
  getline(cin, data);

  for (auto &ch : data) {
    ch = tolower(ch);
  }

  int size = data.length();
  auto start = std::chrono::high_resolution_clock::now();

  char *data_d;
  unsigned int *histo;

  cudaMalloc(&data_d, size * sizeof(char));
  cudaMalloc(&histo, NUM_BINS * sizeof(unsigned int));
  cudaMemset(histo, 0, NUM_BINS * sizeof(unsigned int));

  cudaMemcpy(data_d, data.c_str(), size * sizeof(char), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;

  histo_private_kernel<<<grid_size, block_size>>>(data_d, size, histo);
  cudaDeviceSynchronize();

  cudaMemcpy(count_histo, histo, NUM_BINS * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

  cudaFree(data_d);
  cudaFree(histo);

  printf("THE NUMBER OF OCCURRENCES OF EACH GROUP IS:\n");
  for (int i = 0; i < NUM_BINS; ++i) {
    if (i < 6) {
      printf("%c %c %c %c : %u\n", 'a' + 4 * i, 'b' + 4 * i, 'c' + 4 * i,
             'd' + 4 * i, count_histo[i]);
    } else {
      printf("%c %c      : %u\n", 'a' + 4 * i, 'b' + 4 * i, count_histo[i]);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  cout << "Total execution time: " << duration.count() << " ms\n";
}
