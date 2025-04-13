#include "../utils/cuda_context.cuh"
#include <cstdio>

#define OFFSET(col, x, y) ((col) * (x) + (y))

void rand_matrix(int m, int n, float* M) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            M[OFFSET(n, i, j)] = 2.0 * (float)drand48() - 1.0;
        }
    }
}

void cpu_gemm(int M, int N, int K, float* A, float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[OFFSET(N, i, j)] = 0.0;
            for (int k = 0; k < K; ++k) {
                C[OFFSET(N, i, j)] += A[OFFSET(K, i, k)] * B[OFFSET(N, k, j)];
            }
        }
    }
}

float compare_matrics(int M, int N, float *A, float *B) {
    float max_diff = 0.0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            max_diff = std::max(max_diff, std::abs(A[OFFSET(N, i, j)] - B[OFFSET(N, i, j)]));
        }
    }
    return max_diff;
}

// A: M * K
// B: K * N
__global__ void sgemm(int M, int N, int K, float* A, float* B, float* C) {
    extern __shared__ float smem[];
    int block_size = blockDim.x;
    float* sub_A = smem;  // blockDim.x, blockDim.x
    float* sub_B = sub_A + block_size * block_size;  // blockDim.x, blockDim.x

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;
    for (int s = 0; s < K; s += blockDim.x) {
        if (x < M && s + threadIdx.y < K)
            sub_A[OFFSET(block_size, threadIdx.x, threadIdx.y)] = A[OFFSET(K, x, s + threadIdx.y)];
        if (s + threadIdx.x < K && y < N)
            sub_B[OFFSET(block_size, threadIdx.x, threadIdx.y)] = B[OFFSET(N, s + threadIdx.x, y)];
        __syncthreads();

        if (x < M && y < N) {
            for (int k = 0; k < block_size && s + k < K; ++k) {
                tmp += sub_A[OFFSET(block_size, threadIdx.x, k)] * sub_B[OFFSET(block_size, k, threadIdx.y)];
            }
        }
        __syncthreads();
    }
    if (x < M && y < N) {
        C[OFFSET(N, x, y)] = tmp;
    }
}

void print_matrix(int M, int N, float* A) {
    printf("matrix:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f, ", A[OFFSET(N, i, j)]);
        }
        printf("\n");
    }
}

int main() {
    size_t M = 255;
    size_t N = 257;
    constexpr size_t K = 250;

    float* A, *B, *C, *CPU_C;
    cudaMallocManaged((void**)&A, M * K * sizeof(float));
    cudaMallocManaged((void**)&B, K * N * sizeof(float));
    cudaMallocManaged((void**)&C, M * N * sizeof(float));
    CPU_C = (float*)malloc(M * N * sizeof(float));
    memset(C, 0, sizeof(float) * M * N);
    rand_matrix(M, K, A);
    rand_matrix(K, N, B);

    // print_matrix(M, K, A);
    // print_matrix(K, N, B);

    cpu_gemm(M, N, K, A, B, CPU_C);
    // print_matrix(M, N, CPU_C);

    constexpr size_t BLOCK_SIZE = 16;
    dim3 blk(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    constexpr size_t smem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    sgemm<<<grid, blk, smem_size>>>(M, N, K, A, B, C);
    cudaDeviceSynchronize();
    // print_matrix(M, N, C);

    float max_diff = compare_matrics(M, N, CPU_C, C);
    if (max_diff > 0.5) {
        printf("result error!\n");
    } else {
        printf("check success!\n");
    }

    free(CPU_C);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}