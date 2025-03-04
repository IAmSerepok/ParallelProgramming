#include<iostream>
#include<cuda_runtime.h>

__global__ void kernel ( void ) 
{
  int ID  = blockIdx.x * blockDim.x + threadIdx.x;
 // blockIdx.x номер блока
  //blockDim.x количество потоков в блоке
  //threadIdx.x номер потока в блоке
  
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    printf("identificator, or number of thread is %d\n",ID); 
}

using namespace std;

int main(){
    cout << "Hello from CPU!" << endl;
  
    kernel <<< 1, 10 >>>(); //gpu 

    cudaDeviceSynchronize();
    return 0;
}
