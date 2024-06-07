#include "cuda_runtime.h"
#include "cuda_runtim_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>

#define BLOCK_SIZE 128

void random_init(int* data, int size);
void cpuTest(int* dout, int* din, int size, int n_iter);
__global__ void test_kernelfirst(int* d_out, int* d_in, int size);
__global__ void test_kernelsecond(int* d_out, int* d_in);

int main()
{
        cudaError_t cudaError;
        cudaError = cudaSetDevice(7);
        if(cudaError == cudaSuccess)
        {
            std::cout << "choose success" << std::endl;
        }
        else
        {
            std::cout<< "choose fail" << std::endl;
        }

        const int arraySize = 128*768*128;
        int n_iter = 1;
        int* h_in,h_out;
        cudaMallocHost(&h_in, arraySize * sizeof(int));
        cudaMallocHost(&h_out, 768 * 128 * sizeof(int));
        random_init(h_in, arraySize);

        double dur;
        clock_t start1,end1;
        start1 = clock();
        cpuTest(h_out, h_in, 128, n_iter);
        end1 = clock();
        dur = (double)(end1 - start1);
        printf("cpu time: %f\n", (dur / CLOCKS_PER_SEC));

        int* d_in, d_out;
        cudaMalloc(&d_in, arraySize * sizeof(int));
        cudaMalloc(&d_out, 768 * 128 * sizeof(int));
        cudaMemcpy(d_in, h_in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
        dim3 block1(1,64,16);
        dim3 grid1(1,12,8);
        
        test_kernelfirst<<<grid1,block1>>>(d_out, d_in, 128);

        cudaEvent_t start2, end2;
        cudaEventCreate(&start2);
        cudaEventCreate(&end2);
        cudaEventRecord(start2);
        for(int i = 0; i < n_iter; i++)
        {
            test_kernelfirst<<<grid1,block1>>>(d_out, d_in, 128);
        }
        cudaEventRecord(end2);
        cudaEventSynchronize(end2);
        float msec, sec;
        cudaEventElapsedTime(&msec, start2, end2);
        sec = msec / 1000.0;
        cudaEventDestroy(start2);
        cudaEventDestroy(end2);

        printf("Latency of kernelfirst: %f\n", sec);

        dim3 block2(BLOCK_SIZE,4,2);
        dim3 grid2(1,192,64);
        cudaEvent_t start3, end3;
        cudaEventCreate(&start3);
        cudaEventCreate(&end3);
        cudaEventRecord(start3);
        for(int i = 0; i < n_iter; i++)
        {
            test_kernelsecond<<<grid2,block2>>>(d_out, d_in);
        }
        cudaEventRecord(end3);
        cudaEventSynchronize(end3);
        float msec1, sec1;
        cudaEventElapsedTime(&msec, start3, end3);
        sec1 = msec1 / 1000.0;
        cudaEventDestroy(start3);
        cudaEventDestroy(end3);

        printf("Latency of kernelfirst: %f\n", sec1);

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(h_in);
        cudaFree(h_out);

        return 0;
}
void random_init(int* data, int size)
{
    for (size_t i = 0; i < size; i++)
    {
        /* code */
        data[i] = int(rand() % (10 - 1)) + 1;
    }
    
}

void cpuTest(int* dout, int* din, int size, int n_iter)
{
    for (size_t q = 0; q < n_iter; q++)
    {
        for (size_t i = 0; i < 128; i++)
        {
            for (size_t j = 0; j < 768; j++)
            {
                int temp = din[0 + j * size + i * size * 768];
                for (size_t k = 1; k < size; k++)
                {
                    if (din[k + j * size + i * size * 768] > temp)
                    {
                        temp = din[k + j * size + i * size * 768];
                    }
                    
                }
                
                dout[j + i * 768] = temp;
            }
            
        }
        
    }
    
}

__global__ void test_kernelfirst(int* d_out, int* d_in, int size)
{
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    int tiz = blockDim.z * blockIdx.z + threadIdx.z;
    if(tiy < 768 && tiz < 128)
    {
        int temp = d_in[0 + tiy * 128 + tiz * 768 * 128]
        for (size_t i = 0; i < size; i++)
        {
            if(temp < d_in[i + tiy * 128 + tiz * 768 * 128])
            {
                temp = d_in[i + tiy * 128 + tiz * 768 * 128];
            }
        }
        d_out[tiy + tiz * 768] = temp;
    }
}

__global__ void test_kernelsecond(int* d_out, int* d_in)
{
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    int tiz = blockDim.z * blockIdx.z + threadIdx.z
    sdata[tid] = d_in[tid + tiy * 128 + tiz * 128 * 768];
    __syncthreads();

    if(tiy < 768 && tiz < 128)
    {
        for (size_t i = 0; i < blockDim.x; i*=2)
        {
            if (tid % (2 * i) == 0)
            {
                if (sdata[tid] < sdata[tid + i])
                {
                    sdata[tid] = sdata[tid + i];
                }
            }
            __syncthreads();
        }
        if(tid == 0)
        {
            d_out[tiy + tiz * 768] = sdata[tid];
        }
    }
}