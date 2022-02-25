// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

__device__
void layerMul(float *C, float *A, float *B, unsigned int w) {
    for(unsigned int i=0; i<w ; i++) {
        for(unsigned int j=0; j<w; j++) {
            for(unsigned int k=0; k<w; k++) {
                C[i*w+j] += A[i*w+k]*B[k*w+j];
            }
        }
    }
}

__global__ 
void matrixMulCUDA(float *C, float *A, float *B, unsigned int w) {

    layerMul(C, A, B, w);
    layerMul(A, B, C, w);
    layerMul(B, C, A, w);

    layerMul(C, A, B, w);
    layerMul(A, B, C, w);
    layerMul(B, C, A, w);

    layerMul(C, A, B, w);
    layerMul(A, B, C, w);
    layerMul(B, C, A, w);

    layerMul(C, A, B, w);
    layerMul(A, B, C, w);
    layerMul(B, C, A, w);
    
}

int matrixMultiply(dim3 &dimsA, dim3 &dimsB){
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    cudaMallocManaged(&h_A, mem_size_A);

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    cudaMallocManaged(&h_B, mem_size_B);

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;
    cudaMallocManaged(&h_C, mem_size_C);

    if (h_A == NULL | h_B == NULL | h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate matrix!\n");
        exit(EXIT_FAILURE);
    }

    const float valB = 4.01f;
    constantInit(h_A, size_A, 1.9f);
    constantInit(h_B, size_B, valB);
    constantInit(h_C, size_A, 0.0f);

    // warm up
    matrixMulCUDA<<<1,1>>>(h_C, h_A, h_B, dimsA.x);

    cudaError_t error;
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Execute the kernel
    int nIter = 10100;

    for (int j = 0; j < nIter; j++)
    {
        matrixMulCUDA<<<1,1>>>(h_C, h_A, h_B, dimsA.x);
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);


    // Check for errors: if all the values are evaluated properly on the GPU

    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);

    return EXIT_SUCCESS;

}

int main() {
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    int devID = 0;

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    int w = 64;

    dim3 dimsA(w, w, 1);
    dim3 dimsB(w, w, 1);

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply(dimsA, dimsB);

    exit(matrix_result);
    return 0;
}
