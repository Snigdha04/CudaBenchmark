// System includes
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>
#include <fstream>

// CUDA runtime
#include <cuda_runtime.h>

#define QPS 12

struct profileTime {
    int idx;
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime;
};

std::vector<profileTime> timeVec(QPS); 
std::ofstream outfile;
std::vector<std::chrono::high_resolution_clock::time_point> record_start = std::vector<std::chrono::high_resolution_clock::time_point>(QPS);
std::vector<std::chrono::high_resolution_clock::time_point> record_end = std::vector<std::chrono::high_resolution_clock::time_point>(QPS);
std::vector<std::chrono::high_resolution_clock::time_point> record_sent = std::vector<std::chrono::high_resolution_clock::time_point>(QPS);

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

void checkCudaErrors(cudaError_t status) {
    if(status != cudaSuccess) {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *data)
{
    // Check status of GPU after stream operations are done
    // std::cout << "Callback \n";
    checkCudaErrors(status);
    int* idx = (int*)data;
    // std::cout << *idx << " : idx \n";
    // timeVec
    // checkCudaErrors( cudaEventCreate(&timeVec[*idx].stop) );
    // checkCudaErrors( cudaEventElapsedTime(&timeVec[*idx].elapsedTime, timeVec[*idx].start, timeVec[*idx].stop) );

    auto stop = std::chrono::high_resolution_clock::now();

    auto request_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - record_start[*idx]);
    outfile << request_time.count() << std::endl;

    // std::cout << request_time.count() << std::endl;

    // std::cout << timeVec[*idx].elapsedTime << " : elapsed time \n";

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
    int nIter = QPS;
    int sleep_duration = 1000/QPS; // dutarion in millisec
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    for (int j = 0; j < QPS; j++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_duration));
        timeVec[j].idx = j;
        // checkCudaErrors( cudaEventCreate(&timeVec[j].start) );
        auto start = std::chrono::high_resolution_clock::now();
        record_start[j] = start;
        matrixMulCUDA<<<1,1,0, stream>>>(h_C, h_A, h_B, dimsA.x);
        checkCudaErrors( cudaStreamAddCallback(stream, myStreamCallback, &timeVec[j].idx, 0) );

        // std::cout << "cuda matmul added callback" << j << " \n";
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
    outfile.open("profile_ouput_" + std::to_string(QPS) + ".txt", std::fstream::out);

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
    outfile.close();

    return 0;
}
