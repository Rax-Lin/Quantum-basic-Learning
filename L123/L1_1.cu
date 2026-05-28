#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>

// batch complex multiplication
__global__ void complexMultiplyKernel(const cuFloatComplex* A, const cuFloatComplex* B, cuFloatComplex* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = cuCmulf(A[idx], B[idx]); 
    }
}

__global__ void complexAddKernel(const cuFloatComplex* A, const cuFloatComplex* B, cuFloatComplex* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = cuCaddf(A[idx], B[idx]);
    }
}

__global__ void complexSubtractKernel(const cuFloatComplex* A, const cuFloatComplex* B, cuFloatComplex* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = cuCsubf(A[idx], B[idx]);
    }
}

__global__ void complexDivideKernel(const cuFloatComplex* A, const cuFloatComplex* B, cuFloatComplex* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = cuCdivf(A[idx], B[idx]);
    }
}

int main() {
    const int N = 100000; 
    size_t size = N * sizeof(cuFloatComplex);

    // allocate memory on Host (CPU)
    cuFloatComplex* h_A = (cuFloatComplex*)malloc(size);
    cuFloatComplex* h_B = (cuFloatComplex*)malloc(size);
    cuFloatComplex* h_C = (cuFloatComplex*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = make_cuFloatComplex(2.0f, 3.0f);  // 2 + 3i
        h_B[i] = make_cuFloatComplex(1.0f, -1.0f); // 1 - 1i
    }
    // allocate memory on Device (GPU)
    cuFloatComplex *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy data from Host to Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // start compute time (ms)
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // warm up GPU
    complexAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // start compute
    // +

    cudaEventRecord(start);
    complexAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Total hardware time (+): " << elapsedTime << " ms" << std::endl;
    std::cout << "Result [0]: " << cuCrealf(h_C[0]) << " + " << cuCimagf(h_C[0]) << "i" << std::endl;

    // -
    cudaEventRecord(start);
    complexSubtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Total hardware time (-): " << elapsedTime << " ms" << std::endl;
    std::cout << "Result [0]: " << cuCrealf(h_C[0]) << " + " << cuCimagf(h_C[0]) << "i" << std::endl;
    // *
    cudaEventRecord(start);
    complexMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Total hardware time (*): " << elapsedTime << " ms" << std::endl;
    std::cout << "Result [0]: " << cuCrealf(h_C[0]) << " + " << cuCimagf(h_C[0]) << "i" << std::endl;
    // /
    cudaEventRecord(start);
    complexDivideKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Total hardware time (/): " << elapsedTime << " ms" << std::endl;
    std::cout << "Result [0]: " << cuCrealf(h_C[0]) << " + " << cuCimagf(h_C[0]) << "i" << std::endl;

    // release memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}