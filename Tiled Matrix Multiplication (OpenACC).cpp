#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32  // Define the size of the tile

// Function to initialize matrix with random data
void initializeMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = rand() / (float)RAND_MAX;
        }
    }
}

int main() {
    int M = 512; // Rows in Matrix A and Rows in Resultant Matrix C
    int N = 512; // Columns in Matrix A and Rows in Matrix B
    int P = 512; // Columns in Matrix B and Columns in Resultant Matrix C

    float *A, *B, *C;

    // Allocate memory for matrices
    A = (float *)malloc(M * N * sizeof(float));
    B = (float *)malloc(N * P * sizeof(float));
    C = (float *)malloc(M * P * sizeof(float));

    // Initialize matrices with random values
    initializeMatrix(A, M, N);
    initializeMatrix(B, N, P);

    // OpenACC - Setup data environment and execute the parallel region
    #pragma acc data copyin(A[0:M*N], B[0:N*P]) copyout(C[0:M*P])
    {
        #pragma acc parallel loop
        for (int i = 0; i < M; i += TILE_SIZE) {
            for (int j = 0; j < P; j += TILE_SIZE) {
                #pragma acc loop
                for (int ii = i; ii < i + TILE_SIZE && ii < M; ii++) {
                    for (int jj = j; jj < j + TILE_SIZE and jj < P; jj++) {
                        float sum = 0;
                        for (int k = 0; k < N; k++) {
                            sum += A[ii * N + k] * B[k * P + jj];
                        }
                        C[ii * P + jj] = sum;
                    }
                }
            }
        }

    }

    free(A);
    free(B);
    free(C);

    return 0;
}
