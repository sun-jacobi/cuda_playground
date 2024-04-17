#include <memory>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>


constexpr size_t N = 256; 
constexpr size_t M = 256;
constexpr size_t L = 256;

__global__ void 
matMulKernel(float *A, float *B, float *C) 
{
    int i = threadIdx.x + blockDim.x * blockIdx.x; // col
    int j = threadIdx.y + blockDim.y * blockIdx.y; // row

    if (i < N || j < L) {
        float sum = 0;
        for (size_t k = 0; k < M; k++) 
            sum += A[i * M + k] * B[k * L + j];

        C[i * L + j] = sum;    
    }  
}

void 
matMul(float *A, float *B, float *C) 
{   
    printf("Matrix Addition Kernel\n");  
    //============================
    size_t ASize = N * M * sizeof(float);
    size_t BSize = M * L * sizeof(float); 
    size_t CSize = N * L * sizeof(float); 
    float *DA, *DB, *DC;
    
    cudaMalloc(&DA, ASize);
    cudaMalloc(&DB, BSize);
    cudaMalloc(&DC, CSize);
    
    cudaMemcpy(DA, A, ASize, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, B, BSize, cudaMemcpyHostToDevice);
    
    //============================
    dim3 dimBlock(16, 16);
    dim3 dimGrid(N / dimBlock.x, L / dimBlock.y);
    matMulKernel<<<dimGrid, dimBlock>>>(DA, DB, DC);
    cudaDeviceSynchronize();
    cudaMemcpy(C, DC, CSize, cudaMemcpyDeviceToHost);
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
    {
        for (size_t j = 0; j < L; j++) 
        {
            float element = 0;
            for (size_t k = 0; k < M; k++) 
            {
                element += A[i * M + k] * B[k * L + j];
            }
            err += abs(C[i * L + j] - element);
        }
    }

    printf("A: (%zu, %zu)\n", N, M);
    printf("B: (%zu, %zu)\n", M, L);
    printf("Verification Error: %f\n", err); 
}


int 
main() 
{   
    // A : N * M 
    // B : M * L
    // C : N * L
    float *A = (float *)malloc(N * M * sizeof(float));  
    float *B = (float *)malloc(M * L * sizeof(float));
    float *C = (float *)malloc(N * L * sizeof(float));
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++) 
            A[i * M] = (i * M + j);

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < L; j++) 
            B[i * L] = (i * L + j);
    
        
    matMul(A, B, C);
    verify(A, B, C);
}
