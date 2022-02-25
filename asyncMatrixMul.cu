#include <math.h>
#include <iostream>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);
    //   add<<<1, 256>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);




    // int n = 16 * 1024 * 1024;
    // int nbytes = n * sizeof(int);
    // int value = 26;

    // // allocate host memory
    // int *a = 0;
    // checkCudaErrors(cudaMallocHost((void **)&a, nbytes));
    // memset(a, 0, nbytes);

    // // allocate device memory
    // int *d_a=0;
    // checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
    // checkCudaErrors(cudaMemset(d_a, 255, nbytes));

    // cudaMallocManaged(&a, nbytes);
    // cudaMallocManaged(&d_a, nbytes);

    // // set kernel launch configuration
    // dim3 threads = dim3(512, 1);
    // dim3 blocks  = dim3(n / threads.x, 1);

    // cudaEvent_t start, stop;
    // checkCudaErrors(cudaEventCreate(&start));
    // checkCudaErrors(cudaEventCreate(&stop));

    // // StopWatchInterface *timer = NULL;
    // // sdkCreateTimer(&timer);
    // // sdkResetTimer(&timer);

    // checkCudaErrors(cudaDeviceSynchronize());
    // float gpu_time = 0.0f;

    // // asynchronously issue work to the GPU (all to stream 0)
    // // sdkStartTimer(&timer);
    // cudaEventRecord(start, 0);
    // cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, 0);
    // increment_kernel<<<blocks, threads, 0, 0>>>(d_a, value);
    // cudaMemcpyAsync(a, d_a, nbytes, cudaMemcpyDeviceToHost, 0);
    // cudaEventRecord(stop, 0);
    // // sdkStopTimer(&timer);

    // // have CPU do some work while waiting for stage 1 to finish
    // unsigned long int counter=0;

    // while (cudaEventQuery(stop) == cudaErrorNotReady)
    // {
    //     counter++;
    // }

    // checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

    // // print the cpu and gpu times
    // printf("time spent executing by the GPU: %.2f\n", gpu_time);
    // // printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
    // printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

    // // check the output for correctness
    // // bool bFinalResults = correct_output(a, n, value);

    // // release resources
    // checkCudaErrors(cudaEventDestroy(start));
    // checkCudaErrors(cudaEventDestroy(stop));
    // checkCudaErrors(cudaFreeHost(a));
    // checkCudaErrors(cudaFree(d_a));

    // // cudaDeviceReset causes the driver to clean up all state. While
    // // not mandatory in normal operation, it is good practice.  It is also
    // // needed to ensure correct operation when the application is being
    // // profiled. Calling cudaDeviceReset causes all profile data to be
    // // flushed before the application exits
    // cudaDeviceReset();
  
  return 0;
}