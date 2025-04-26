#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 32  // matrix is N x N
#define BINS 10

__global__ void histogram2d_kernel(int *mat, int *hist) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) {
        int val = mat[row * N + col];
        if(val >= 1 && val <= 10) {
            atomicAdd(&hist[val - 1], 1);
        }
    }
}

int main() {
    int *mat_h, *mat_d, *hist_h, *hist_d;

    mat_h = (int*) malloc(N * N * sizeof(int));
    hist_h = (int*) calloc(BINS, sizeof(int));

    // Fill matrix with random values between 1 and 10
    for(int i = 0; i < N * N; i++) {
        mat_h[i] = (rand() % 10) + 1;
    }

    cudaMalloc(&mat_d, N * N * sizeof(int));
    cudaMalloc(&hist_d, BINS * sizeof(int));
    cudaMemcpy(mat_d, mat_h, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(hist_d, hist_h, BINS * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    histogram2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(mat_d, hist_d);
    cudaDeviceSynchronize();

    cudaMemcpy(hist_h, hist_d, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    printf("2D Histogram (bins 1 to 10):\n");
    for(int i = 0; i < BINS; i++) {
        printf("Bin %2d: %d\n", i + 1, hist_h[i]);
    }

    cudaFree(mat_d);
    cudaFree(hist_d);
    free(mat_h);
    free(hist_h);

    return 0;
}

