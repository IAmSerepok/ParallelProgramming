#include <iostream>
#include <cuda_runtime.h>

#define N 10000

__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = i;
    }
   
    float *dev_A, *dev_B, *dev_C;
    cudaMalloc((void**)&dev_A, sizeof(float) * N);
    cudaMalloc((void**)&dev_B, sizeof(float) * N);
    cudaMalloc((void**)&dev_C, sizeof(float) * N);

    cudaMemcpy(dev_A, A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Создание событий для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запись времени начала
    cudaEventRecord(start, 0);

    // Запуск ядра
    VecAdd<<<(N + 255) / 256, 256>>>(dev_A, dev_B, dev_C);
    cudaDeviceSynchronize();

    // Запись времени окончания
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Получение времени выполнения
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(C, dev_C, sizeof(float) * N, cudaMemcpyDeviceToHost);

    std::cout << "C[1000]: " << C[1000] << " C[1001]: " << C[1001] << std::endl;
    std::cout << "Время выполнения на GPU: " << milliseconds << " мс" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    // Освобождение событий
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
