#include <memory>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// A shared memory version of matrix multiplication

constexpr size_t N = 512; 
constexpr size_t M = 256;
constexpr size_t L = 512;
constexpr size_t BLOCK_SIZE = 16;

__global__ void 
matMulKernel(float *A, float *B, float *C) 
{   
    // Block index 
    size_t BI = blockIdx.x;
    size_t BJ = blockIdx.y;

    float SUM = 0;

    // Thread index in Block
    size_t I = threadIdx.x;
    size_t J = threadIdx.y;

    for (size_t BK = 0; BK < (M / BLOCK_SIZE); BK++) {
        __shared__ float SA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float SB[BLOCK_SIZE][BLOCK_SIZE];
        // load global memory into shared memory
        SA[I][J] = A[BI * BLOCK_SIZE * M + BK * BLOCK_SIZE + I * M + J];
        SB[I][J] = B[BK * BLOCK_SIZE * L + BJ * BLOCK_SIZE + I * L + J];
        __syncthreads();
        for (int K = 0; K < BLOCK_SIZE; K++)
            SUM += SA[I][K] * SB[K][J];
        __syncthreads();
    }
    // C[i][j] <- SUM
    C[BI * BLOCK_SIZE * L + BJ * BLOCK_SIZE + I * L + J] = SUM;
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
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
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
            A[i * M] = sin(i * M + j);

    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < L; j++) 
            B[i * L] = cos(i * L + j);
         
    matMul(A, B, C);
    verify(A, B, C);
}
