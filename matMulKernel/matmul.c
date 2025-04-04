#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
int **create_matrix(int rows, int cols) {
  int **matrix = (int **)malloc(rows * sizeof(int *));
  for (int i = 0; i < rows; ++i) {
    matrix[i] = (int *)malloc(cols * (sizeof(int)));
  }
  return matrix;
}
void populate(int **matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i][j] = rand() % 100;
    }
  }
}
void print_matrix(int **matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ", matrix[i][j]);
    }
    printf("\n");
  }
}
int **matrix_mul(int **matrix_1, int **matrix_2, int rows, int cols) {
  int **product_matrix = (int **)malloc(rows * (sizeof(int *)));
  for (int i = 0; i < rows; ++i) {
    product_matrix[i] = (int *)malloc(cols * (sizeof(int)));
  }
  printf("Multiplying Matrix\n");
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      int sum = 0;
      for (int k = 0; k < rows; k++) {
        sum += matrix_1[i][k] * matrix_2[k][j];
      }
      product_matrix[i][j] = sum;
    }
  }
  return product_matrix;
}
int main(int argc, char *argv[]) {
  int **matrix_1 = create_matrix(3, 3);
  int **matrix_2 = create_matrix(3, 3);
  populate(matrix_1, 3, 3);
  populate(matrix_2, 3, 3);
  printf("THE FIRST MATRIX IS\n");
  print_matrix(matrix_1, 3, 3);
  printf("THE SECOND MATRIX IS\n");
  print_matrix(matrix_2, 3, 3);
  int **matrix_3 = matrix_mul(matrix_1, matrix_2, 3, 3);
  print_matrix(matrix_3, 3, 3);

  return EXIT_SUCCESS;
}
