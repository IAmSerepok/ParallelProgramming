#include <iostream>
#include <cuda_runtime.h>

using namespace std;

const int N = 2; // Количество строк
const int M = 3; // Количество столбцов

// Kernel definition
__global__ void MatAdd(float A[N][M], float B[N][M], float *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Индекс потока
    if (idx < N * M) {
        int i = idx / M;   // Получаем номер строки
        int j = idx % M;   // Получаем номер столбца
        atomicAdd(result, A[i][j] * B[i][j]); // Скалярное произведение
    }
}

int main() {
    // Выделение двумерного массива
    float (*A)[M] = new float[N][M]; // матрица A
    float (*B)[M] = new float[N][M]; // матрица B
    float h_result = 0.0f; // Результат на хосте

    // Инициализация массивов
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            A[i][j] = static_cast<float>(i * M + j + 1); 
            B[i][j] = static_cast<float>(i * M + j + 1);
        }
    }

    float (*dev_A)[M], (*dev_B)[M]; // Указатели на массивы на устройстве
    float *dev_result; // Указатель на результат на устройстве

    // Выделение памяти на устройстве
    cudaMalloc((void**)&dev_A, N * M * sizeof(float));
    cudaMalloc((void**)&dev_B, N * M * sizeof(float));
    cudaMalloc((void**)&dev_result, sizeof(float)); // Для итоговой суммы

    // Копирование данных на устройство
    cudaMemcpy(dev_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_result, &h_result, sizeof(float), cudaMemcpyHostToDevice); // Сброс итогового результата

    // Создание событий для замера времени
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Запись времени перед запуском ядра
    cudaEventRecord(startEvent, 0);

    // Запуск ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * M + threadsPerBlock - 1) / threadsPerBlock;
    MatAdd<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_result);

    // Копирование результата обратно на хост
    cudaMemcpy(&h_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            cout << A[i][j] << ' ';
        }
        cout << '\n';
    }

    // Вывод результатов
    cout << "Скалярное произведение: " << h_result << endl;

    // Запись времени после завершения ядра
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent); // Ожидание завершения события

    // Получение времени выполнения ядра
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent); // Время в миллисекундах

    // Вывод результатов
    cout << "Время выполнения: " << elapsedTime << " миллисекунд" << endl;

    // Освобождение ресурсов
    delete[] A;
    delete[] B;
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_result);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
