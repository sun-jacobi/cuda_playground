#include <memory>
#include <cmath>
#include <cstdio>

constexpr size_t N = 256;

__global__ void 
matAddKernel(float *A, float *B, float *C) 
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (i < N && j < N) 
       C[i][j] = A[i][j] + B[i][j];
}

void 
matAdd(float *A, float *B, float *C) 
{   
    printf("Matrix Addition Kernel\n");  
    //============================
    size_t SIZE = N * N * sizeof(float);
    float *DA, *DB, *DC;
    
    cudaMalloc(&DA, SIZE);
    cudaMalloc(&DB, SIZE);
    cudaMalloc(&DC, SIZE);
    
    cudaMemcpy(DA, A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B, SIZE, cudaMemcpyHostToDevice);
    
    //============================
    dim3 dimBlock(16, 16);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    matMulKernel<<<dimGrid, dimBlock>>>(DA, DB, DC);
    cudaDeviceSynchronize();
    cudaMemcpy(C, DC, SIZE, cudaMemcpyDeviceToHost);
    //===============
    cudaFree(DA); 
    cudaFree(DB);
    cudaFree(DC);
}

void
verify(float *A, float *B, float *C)
{
    float err = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) 
            err += abs(C[i * N + j] - (A[i * N + j] + B[i * N + j]));
    printf("Matrix Size: (%zu, %zu)\n", N, N);
    printf("Verification Error: %f\n", err); 
}

int 
main() 
{ 
    size_t SIZE = N * N * sizeof(float);
    float *A = (float *)malloc(SIZE);  
    float *B = (float *)malloc(SIZE);
    float *C = (float *)malloc(SIZE);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i * N + j] = sin(i * N + j);
            B[i * N + j] = cos(i * N + j);
        }
    }

    matAdd(A, B, C);
    verify(A, B, C);
}