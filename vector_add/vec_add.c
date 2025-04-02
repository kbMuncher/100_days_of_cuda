#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void vec_add(float *A_h, float *B_h, float *C_h, int n) {
  for (int i = 0; i < n; ++i) {
    C_h[i] = A_h[i] + B_h[i];
  }
}

int main(int argc, char *argv[]) {
  float *A, *B, *C;
  A = (float *)malloc(sizeof(float) * 1000000);
  B = (float *)malloc(sizeof(float) * 1000000);
  C = (float *)malloc(sizeof(float) * 1000000);

  for (int i = 0; i < 1000000; ++i) {
    A[i] = i + 1;
    B[i] = i + 1000001;
  }
  clock_t start, end;
  double time_taken;
  start = clock();
  vec_add(A, B, C, 1000000);
  end = clock();
  time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("Time taken by CPU:- %f\n", time_taken);
  free(A);
  free(B);
  free(C);
  return EXIT_SUCCESS;
}
