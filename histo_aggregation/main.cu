#include <cctype>
#include <chrono>
#include <iostream>
using namespace std;
#define NUM_BINS 7

__global__ void histo_aggregate_kernel(char *data, int length,
                                       unsigned int *histo) {
  // Shared memory of bins
  __shared__ unsigned int histo_p[NUM_BINS];
  for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
    histo_p[i] = 0u;
  }

  // create the histogram
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int pre_IDX =
      -1; // this will be storing the index of element just beforte the current
          // one that will be used to update the accumulattor
  int accumulattor = 0;
  for (unsigned int i = tid; i < length; i += gridDim.x * blockDim.x) {
    int alpha_pos = data[i] - 'a'; // this will define the index
    if (alpha_pos >= 0 & alpha_pos < 26) {
      int bin = alpha_pos / 4; // since we are storing 4 alphabets in 7 bins
      if (bin == pre_IDX)      // the logic of previous comment
      {
        ++accumulattor; // this too
      } else {
        if (accumulattor >
            0) { // signifies that similar index are over and new element is
                 // here that doesn't belong to that bin
          atomicAdd(&histo_p[pre_IDX],
                    accumulattor); // update once for all similar elements
        }
        accumulattor = 1; // not zero since we have to account for new element
                          // that just broke the chain
        pre_IDX = bin;    // the index of this new element
      }
    }
  }
  if (accumulattor > 0) {
    atomicAdd(&histo_p[pre_IDX], accumulattor); // for trailing element/s
  }
  __syncthreads();

  // add to share memory
  for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
    unsigned int binVal = histo_p[bin];
    if (binVal > 0) {
      atomicAdd(&histo[bin], binVal);
    }
  }
}

int main() {
  string data;
  getline(cin, data);
  for (auto &a : data) {
    a = tolower(a);
  }
  int size = data.length();
  unsigned int count_histo[NUM_BINS];

  auto start = std::chrono::high_resolution_clock::now();
  char *data_d;
  unsigned int *histo;

  cudaMalloc(&data_d, size * sizeof(char));
  cudaMalloc(&histo, NUM_BINS * sizeof(unsigned int));
  cudaMemset(histo, 0, NUM_BINS * sizeof(unsigned int));

  cudaMemcpy(data_d, data.c_str(), size * sizeof(char), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;

  histo_aggregate_kernel<<<grid_size, block_size>>>(data_d, size, histo);
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
