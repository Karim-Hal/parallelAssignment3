#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 4


__global__ void tiledMatMulKernel(float* A, float* B, float* C, int m, int n, int k) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bsx = blockDim.x;
    int bsy = blockDim.y;

    __shared__ float ds_A[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_B[TILE_SIZE][TILE_SIZE];

    int yid = by * bsy + ty;
    int xid = bx * bsx + tx;

    float sum = 0.0f;

    for (int t = 0; t < (n - 1) / TILE_SIZE + 1; t++) {
        if (yid < m && t * TILE_SIZE + tx < n) ds_A[ty][tx] = A[yid * n + t * TILE_SIZE + tx];
        else ds_A[ty][tx] = 0.0f;

        if (xid < k && t * TILE_SIZE + ty < n) ds_B[ty][tx] = B[(t * TILE_SIZE + ty) * k + xid];
        else ds_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) sum += ds_A[ty][i] * ds_B[i][tx];

        __syncthreads();
    }

    if (yid < m && xid < k) C[yid * k + xid] = sum;
}

int main() {
    int m = 1000;
int n = 1000;
int k = 1000;


    size_t a_size = m * n * sizeof(float);
    size_t b_size = n * k * sizeof(float);
    size_t c_size = m * k * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, a_size);
    cudaMallocManaged(&B, b_size);
    cudaMallocManaged(&C, c_size);

    for (int q = 0; q < m * n; ++q) {
        A[q] = rand() / (float)RAND_MAX;
    }
    for (int w = 0; w < n * k; ++w) {
        B[w] = rand() / (float)RAND_MAX;
    }

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((k - 1) / dimBlock.x + 1, (m - 1) / dimBlock.y + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    tiledMatMulKernel<<<dimGrid, dimBlock>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("The elapsed time is: %f", milliseconds);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

   

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}