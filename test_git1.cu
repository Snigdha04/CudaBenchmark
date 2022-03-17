#include <stdio.h>
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



// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  float x = (float)i;
  float s = sinf(x); 
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s*s+c*c);
}

float maxError(float *a, int n) 
{
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 400;
  const long int n = 4 * 1024 * blockSize * nStreams;
  const long int streamSize = n / nStreams;
  const long int streamBytes = streamSize * sizeof(float);
  const long int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );
  
  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

  float ms; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );

  auto start_prof = std::chrono::high_resolution_clock::now();
  auto end_prof = std::chrono::high_resolution_clock::now();
  auto request_prof = std::chrono::duration_cast<std::chrono::microseconds>(end_prof-start_prof);

  
  // baseline case - sequential transfer and execute
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );

  start_prof = std::chrono::high_resolution_clock::now();
  checkCuda( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );
  end_prof = std::chrono::high_resolution_clock::now();
  request_prof = std::chrono::duration_cast<std::chrono::microseconds>(end_prof-start_prof);
  std::cout << request_prof.count() << "us : cudaMemcpy duration1\n";

  kernel<<<n/blockSize, blockSize>>>(d_a, 0);

  start_prof = std::chrono::high_resolution_clock::now();
  checkCuda( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
  end_prof = std::chrono::high_resolution_clock::now();
  request_prof = std::chrono::duration_cast<std::chrono::microseconds>(end_prof-start_prof);
  std::cout << request_prof.count() << "us : cudaMemcpy duration2\n";

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 1: loop over {copy, kernel, copy}
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );

  
  for (int i = 0; i < nStreams; ++i) {
    
    int offset = i * streamSize;

    start_prof = std::chrono::high_resolution_clock::now();
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice, 
                               stream[i]) );

    end_prof = std::chrono::high_resolution_clock::now();
    request_prof = std::chrono::duration_cast<std::chrono::microseconds>(end_prof-start_prof);
    std::cout << request_prof.count() << "us : cudaMemcpyAsync duration version 1\n";


    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }

  

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // asynchronous version 2: 
  // loop over copy, loop over kernel, loop over copy
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );

  start_prof = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
  }

  end_prof = std::chrono::high_resolution_clock::now();
  request_prof = std::chrono::duration_cast<std::chrono::microseconds>(end_prof-start_prof);
  std::cout << request_prof.count() << "us : cudaMemcpyAsync duration1 version 2\n";


  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }

  start_prof = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }

  end_prof = std::chrono::high_resolution_clock::now();
  request_prof = std::chrono::duration_cast<std::chrono::microseconds>(end_prof-start_prof);
  std::cout << request_prof.count() << "us : cudaMemcpyAsync duration2 version 2\n";

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}