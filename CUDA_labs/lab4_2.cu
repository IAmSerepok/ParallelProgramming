#include <iostream>
#include <cuda_runtime.h>

using namespace std;

const int N = 2; // Количество строк
const int M = 3; // Количество столбцов

// Kernel definition
__global__ void MatAdd(float A[N * M], float B[N * M], float C[N * M]) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Индекс потока
    if (idx < N * M) {
        C[idx] = A[idx] * B[idx]; // Скалярное произведение
    }
}


// Функция для редукции массива до суммы
__global__ void reduce(float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    // Индекс текущего потока
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Загружаем данные в shared memory
    sdata[tid] = (i < n) ? g_idata[i] : 0; 
    __syncthreads();

    // Редукция
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Записываем результат в output
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


int main() {
    float *A = new float[N * M]; // матрица A
    float *B = new float[N * M]; // матрица B
    float *C = new float[N * M]; // матрица C

    // Инициализация массивов
    for (int i = 0; i < N * M; ++i) {
        A[i] = static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 1);
        cout << A[i] << " ";
    }

    float *dev_A, *dev_B, *dev_C, *dev_sum; 
    int numBlocks = (N * M + 255) / 256;

    // Выделение памяти на устройстве
    cudaMalloc(&dev_A, N * M * sizeof(float));
    cudaMalloc(&dev_B, N * M * sizeof(float));
    cudaMalloc(&dev_C, N * M * sizeof(float));
    cudaMalloc(&dev_sum, numBlocks * sizeof(float));

    // Копирование данных на устройство
    cudaMemcpy(dev_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Создание событий для замера времени
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Запись времени перед запуском ядра
    cudaEventRecord(startEvent, 0);

    // Запуск ядра
    MatAdd<<<numBlocks, 256>>>(dev_A, dev_B, dev_C);
    cudaDeviceSynchronize();

    reduce<<<numBlocks, 256, 256 * sizeof(int)>>>(dev_C, dev_sum, N * M);
    cudaDeviceSynchronize();

    float finalSum;
    float *h_sum = (float *)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_sum, dev_sum, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Подсчитываем финальную сумму на CPU
    finalSum = 0;
    for (int i = 0; i < numBlocks; i++) {
        finalSum += h_sum[i];
    }

    // Вывод результата
    printf("Final sum is: %f\n", finalSum);

    // Запись времени после завершения ядра
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent); // Ожидание завершения события

    // Получение времени выполнения ядра
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent); // Время в миллисекундах

    // Вывод результатов
    cout << "Время выполнения: " << elapsedTime << " миллисекунд" << endl;

    // Освобождение ресурсов
    free(h_sum);
    cudaFree(dev_sum);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
