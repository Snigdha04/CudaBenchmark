// System includes
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>
#include <fstream>
// #include <torch/torch.h>

// CUDA runtime
#include <cuda_runtime.h>

#define QPS 2000

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

void constantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
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

template <int BLOCK_SIZE> __device__ 
void layerMulTile(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

template <int BLOCK_SIZE> __global__ 
void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {

    layerMulTile<BLOCK_SIZE>(C, A, B, wA, wB);
    layerMulTile<BLOCK_SIZE>(A, B, C, wA, wB);
    layerMulTile<BLOCK_SIZE>(B, C, A, wA, wB);

    layerMulTile<BLOCK_SIZE>(C, A, B, wA, wB);
    layerMulTile<BLOCK_SIZE>(A, B, C, wA, wB);
    layerMulTile<BLOCK_SIZE>(B, C, A, wA, wB);

    layerMulTile<BLOCK_SIZE>(C, A, B, wA, wB);
    layerMulTile<BLOCK_SIZE>(A, B, C, wA, wB);
    layerMulTile<BLOCK_SIZE>(B, C, A, wA, wB);

    layerMulTile<BLOCK_SIZE>(C, A, B, wA, wB);
    layerMulTile<BLOCK_SIZE>(A, B, C, wA, wB);
    layerMulTile<BLOCK_SIZE>(B, C, A, wA, wB);
    
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

int matrixMultiply_include_A_memcpy(dim3 &dimsA, dim3 &dimsB, int block_size){

    // Allocate host memory for matrices A, B and C
    unsigned int image_cnt = QPS;
    std::vector<float *> images;

    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    for (int i=0; i<image_cnt; i++) {
        h_A == NULL;
        h_A = (float *)malloc(mem_size_A);
        images.push_back(h_A);
        if (h_A == NULL)
        {
            fprintf(stderr, "Failed to allocate matrix!\n");
            exit(EXIT_FAILURE);
        }
    }
    
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_B == NULL | h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate matrix!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host memory
    const float valB = 0.01f;
    for (int i=0; i<image_cnt; i++) {
        constantInit(images[i], size_A, 1.0f);
    }
    // constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);
    constantInit(h_C, size_A, 0.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaErrors( cudaMalloc((void **) &d_A, mem_size_A) );
    checkCudaErrors( cudaMalloc((void **) &d_B, mem_size_B) );
    checkCudaErrors( cudaMalloc((void **) &d_C, mem_size_C) );

    // copy host memory to device
    // checkCudaErrors( cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // warm up just for matrix B - the weight matrix!
    for(int i=0; i<10; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        matrixMulCUDA<32><<< grid,threads >>>(d_C, d_B, d_B, dimsA.x, dimsB.x);
    }
    
    cudaDeviceSynchronize();

    cudaEvent_t start;
    checkCudaErrors( cudaEventCreate(&start) );

    cudaEvent_t stop;
    checkCudaErrors( cudaEventCreate(&stop) );

    // Record the start event
    checkCudaErrors( cudaEventRecord(start, NULL) );

    // Execute the kernel
    int nIter = QPS;
    int sleep_duration = (1000*1000)/QPS; // dutarion in microseconds
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    auto start_exp = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < QPS; j++) {
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration));
        timeVec[j].idx = j;
        // checkCudaErrors( cudaEventCreate(&timeVec[j].start) );
        auto start_time = std::chrono::high_resolution_clock::now();
        record_start[j] = start_time;

        checkCudaErrors( cudaMemcpyAsync(d_A, images[j], mem_size_A, cudaMemcpyHostToDevice, stream) );

        matrixMulCUDA<32><<< grid,threads,0, stream >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

        checkCudaErrors( cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream) );

        checkCudaErrors( cudaStreamAddCallback(stream, myStreamCallback, &timeVec[j].idx, 0) );
        // std::cout << "cuda matmul added callback" << j << " \n";
    }

    auto end_exp = std::chrono::high_resolution_clock::now();

    auto request_exp = std::chrono::duration_cast<std::chrono::microseconds>(end_exp-start_exp);
    std::cout << request_exp.count() << ": is the experiment duration\n";

    // Record the stop event
    checkCudaErrors( cudaEventRecord(stop, NULL) );

    

    // Wait for the stop event to complete
    checkCudaErrors( cudaEventSynchronize(stop) );

    float msecTotal = 0.0f;
    checkCudaErrors( cudaEventElapsedTime(&msecTotal, start, stop) );

    // float msecTotal = 0.0f;
    // checkCudaErrors( cudaEventElapsedTime(&msecTotal, start, stop) );

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        gigaFlops,
        msecPerMatrixMul,
        flopsPerMatrixMul);

    checkCudaErrors( cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    return EXIT_SUCCESS;

}

int matrixMultiply(dim3 &dimsA, dim3 &dimsB, int block_size){
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

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // warm up
    for(int i=0; i<10; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        matrixMulCUDA<32><<< grid,threads >>>(h_C, h_A, h_B, dimsA.x, dimsB.x);
    }
    
    cudaDeviceSynchronize();

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
        matrixMulCUDA<32><<< grid,threads,0, stream >>>(h_C, h_A, h_B, dimsA.x, dimsB.x);
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

    // int w = 64;
    // dim3 dimsA(w, w, 1);
    // dim3 dimsB(w, w, 1);

    int block_size = 32;
    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*2*block_size, 5*2*block_size, 1);

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    int matrix_result = matrixMultiply_include_A_memcpy(dimsA, dimsB, block_size);

    exit(matrix_result);
    outfile.close();

    return 0;
}
