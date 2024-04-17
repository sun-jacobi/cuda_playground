#include <memory>
#include <cmath>
#include <cstdio>

constexpr size_t N = 256;

__global__ void 
matAddKernel(float A[N][N], float B[N][N], float C[N][N]) 
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (i < N && j < N) 
       C[i][j] = A[i][j] + B[i][j];
}

void 
matAdd(float A[N][N], float B[N][N], float C[N][N]) 
{   
    printf("Matrix Addition Kernel\n");  
    //============================
    int size = N * N * sizeof(float); 
    float (*DA)[N], (*DB)[N], (*DC)[N];
    cudaMalloc(&DA, size); cudaMemcpy(DA, A, size, cudaMemcpyHostToDevice);
    cudaMalloc(&DB, size); cudaMemcpy(DB, B, size, cudaMemcpyHostToDevice);
    cudaMalloc(&DC, size);
    
    //============================
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    matAddKernel<<<numBlocks, threadsPerBlock>>>(DA, DB, DC);
    cudaMemcpy(C, DC, size, cudaMemcpyDeviceToHost);
    //===============
    cudaFree(DA); 
    cudaFree(DB);
    cudaFree(DC);
}

void
verify(float A[N][N], float B[N][N], float C[N][N])
{
    float err = 0;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) 
            err += abs(C[i][j] - (A[i][j] + B[i][j]));
    printf("Matrix Size: (%zu, %zu)\n", N, N);
    printf("Verification Error: %f\n", err); 
}

int 
main() 
{
    float (*A)[N] = (float (*)[N])malloc(N * N * sizeof(float));  
    float (*B)[N] = (float (*)[N])malloc(N * N * sizeof(float));
    float (*C)[N] = (float (*)[N])malloc(N * N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            A[i][j] = sin(i * N + j);
            B[i][j] = cos(i * N + j);
        }
    }
    matAdd(A, B, C);
    verify(A, B, C);
}