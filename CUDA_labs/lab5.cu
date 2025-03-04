#include <iostream>
#include <cuda_runtime.h>

using namespace std;

const int N = 16;
const int KERNEL = 16;

__global__ void transpose(float *odata, const float *idata, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        int inIndex = x + n * y;
        int outIndex = y + n * x;
        odata[outIndex] = idata[inIndex];
    }
}

int main() {
    float (*A)[N] = new float[N][N];
    float (*B)[N] = new float[N][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = (i * N + j + 1) * 1.0;  
        }
    }

    float *dev_A, *dev_B;

    cudaMalloc((void**)&dev_A, N * N * sizeof(float));
    cudaMalloc((void**)&dev_B, N * N * sizeof(float));

    cudaMemcpy(dev_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    dim3 threads(KERNEL, KERNEL);
    dim3 grid((N + KERNEL - 1) / KERNEL, (N + KERNEL - 1) / KERNEL);

    transpose<<<grid, threads>>>(dev_B, dev_A, N);

    cudaMemcpy(B, dev_B, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    cout << "Время выполнения: " << elapsedTime << " миллисекунд" << endl;

    cout << "Исходная матрица A:" << endl;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            cout << A[i][j] << ' ';
        }
        cout << endl;
    }

    cout << "Транспонированная матрица B:" << endl;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            cout << B[i][j] << ' ';
        }
        cout << endl;
    }

    delete[] A;
    delete[] B;
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
