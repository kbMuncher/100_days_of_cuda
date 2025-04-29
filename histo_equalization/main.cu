#include <cmath>
#include <cstdio>
#include <cstdlib>
#define SIZE 5
void pt_mt(int *matrix) {
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      printf("%d ", matrix[i * SIZE + j]);
    }
    printf("\n");
  }
}
int main() {
  int *matrix = new int[SIZE * SIZE];
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      matrix[i * SIZE + j] = rand() % 10 + 1;
    }
  }
  pt_mt(matrix);
  printf("\n");
  // counting func
  int *count = new int[10];
  float prob[10];
  for (int i = 0; i < 10; ++i) {
    count[i] = 0;
  }

  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      count[matrix[i * SIZE + j] - 1]++;
    }
  }

  printf("The counts of individual elements are \n");

  for (int i = 0; i < 10; ++i) {
    printf("%d ", count[i]);
  }
  printf("\n");

  // probab
  printf("The probs are\n");
  for (int i = 0; i < 10; ++i) {
    prob[i] = static_cast<float>(count[i]) / (SIZE * SIZE);
    printf("%.3f ", prob[i]);
  }
  printf("\n");
  // CDF
  printf("The CDFprobs are\n");
  for (int i = 0; i < 10; ++i) {
    if (i > 0) {

      prob[i] = prob[i - 1] + prob[i];
    }
    printf("%.3f ", prob[i]);
  }
  printf("\n");
  // (L-1)*CDF
  printf("After Scaling\n");
  for (int i = 0; i < 10; ++i) {
    prob[i] = 10 * prob[i];
    printf("%.4f ", prob[i]);
  }
  printf("\n");
  // new vals
  int n_c[10];
  printf("NEW VALS\n");
  for (int i = 0; i < 10; ++i) {
    n_c[i] = static_cast<int>(round(prob[i]));
    printf("%d ", n_c[i]);
  }
  printf("\n");

  // new matrix
  printf("\n");
  printf("RECONSTRUCTED MATRIX\n");
  for (int i = 0; i < SIZE; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      matrix[i * SIZE + j] = n_c[matrix[i * SIZE + j] - 1];
      printf("%d ", matrix[i * SIZE + j]);
    }
    printf("\n");
  }
}
