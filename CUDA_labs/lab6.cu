#include <iostream>
#include <cuda_runtime.h>

using namespace std;

const int N = 4; 
const int KERNEL = 2; 

__global__ void matMul(float *a, float *b, int n, float *c) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0;

    int ia = n * (KERNEL * by + ty);
    int ib = KERNEL * bx + tx;

    for (int k = 0; k < n; ++k) {
        sum += a[ia + k] * b[k * n + ib];
    }

    int ic = n * (KERNEL * by + ty) + (KERNEL * bx + tx);
    c[ic] = sum;
}

int main() {
    float (*A)[N] = new float[N][N];
    float (*B)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) { 
            A[i][j] = (i + j) * 1.0;
            B[i][j] = (i + j) * 1.0;
            cout << (i + j) * 1.0 << " ";
        }
        cout << "\n";
    }

    float *dev_A, *dev_B, *dev_C;
    cudaMalloc((void**)&dev_A, N * N * sizeof(float));
    cudaMalloc((void**)&dev_B, N * N * sizeof(float));
    cudaMalloc((void**)&dev_C, N * N * sizeof(float));

    cudaMemcpy(dev_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);

    dim3 threads(KERNEL, KERNEL);
    dim3 grid((N + KERNEL - 1) / KERNEL, (N + KERNEL - 1) / KERNEL);

    matMul<<<grid, threads>>>(dev_A, dev_B, N, dev_C);

    cudaMemcpy(C, dev_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    cout << "Время выполнения: " << elapsedTime << " миллисекунд" << endl;

    cout << "Результат матричного умножения C:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << C[i][j] << ' ';
        }
        cout << endl;
    }

    delete[] A;
    delete[] B;
    delete[] C; 
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
