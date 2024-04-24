#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define block_size 4


__global__ void MatMulKernel(float* A, float* B, float* C, int m, int n, int k) {
    int yid = (blockIdx.y * block_size) + threadIdx.y;
    int xid = (blockIdx.x * block_size) + threadIdx.x;
    // make sure we're still within bounds
    if ((yid < m) && (xid < k)) {
        float temp =0;
        for (int i = 0; i < n; ++i) {
            temp += A[(yid * n) + i] * B[(i * k) + xid];
        }
        C[(yid * k) + xid] = temp;
    }
}

int main() {
    // initialize matrix dimensions
    int m{1000};
    int n{1000};
    int k{1000};

    size_t a_size = m * n * sizeof(float);
    size_t b_size = n * k * sizeof(float);
    size_t c_size = m * k * sizeof(float);

    // allocate unified memory for the matrices
    float *A, *B, *C;
    cudaMallocManaged(&A, a_size);
    cudaMallocManaged(&B, b_size);
    cudaMallocManaged(&C, c_size);

    // initialize matrices A and B with random values
    for (int q = 0; q < m * n; ++q) {
        A[q] = rand() / (float)RAND_MAX;
    }
    for (int w = 0; w < n * k; ++w) {
        B[w] = rand() / (float)RAND_MAX;
    }

    // define block and grid dimensions

    
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((k - 1) / dimBlock.x + 1, (m - 1) / dimBlock.y + 1);

    // Use CUDA to track the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    // call the matrix multiplication kernel
    MatMulKernel<<<dimGrid, dimBlock>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();

    // stop tracking the time
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // report the kernel execution time
   printf("The elapsed time is: %f", milliseconds);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

   

    // free the allocated memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}