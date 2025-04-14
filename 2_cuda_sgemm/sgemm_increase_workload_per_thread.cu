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
template<unsigned int THREAD_SCALE>
__global__ void sgemm(int M, int N, int K, float* A, float* B, float* C) {
    extern __shared__ float smem[];
    int thread_block_size = blockDim.x;
    int data_block_size = thread_block_size * THREAD_SCALE;
    float* sub_A = smem;  // blockDim.x, blockDim.x
    float* sub_B = sub_A + data_block_size * data_block_size;  // blockDim.x, blockDim.x

    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp_res[THREAD_SCALE][THREAD_SCALE] = {0};
    int slide_x = blockIdx.x * data_block_size;
    int slide_y = blockIdx.y * data_block_size;
    for (int s = 0; s < K; s += data_block_size) {
        // (slide_x, s) (slide_x + data_block_size, s + data_block_size)
        for (int x_abs_offset = slide_x + threadIdx.x; x_abs_offset < M && x_abs_offset < slide_x + data_block_size; x_abs_offset += thread_block_size) {
            for (int y_abs_offset = s + threadIdx.y; y_abs_offset < K && y_abs_offset < s + data_block_size; y_abs_offset += thread_block_size) {
                sub_A[OFFSET(data_block_size, x_abs_offset - slide_x, y_abs_offset - s)] = A[OFFSET(K, x_abs_offset, y_abs_offset)];
            }
        }
        // (s, slide_y) (s + data_block_size, slide_y + data_block_size)
        for (int x_abs_offset = s + threadIdx.x; x_abs_offset < K && x_abs_offset < s + data_block_size; x_abs_offset += thread_block_size) {
            for (int y_abs_offset = slide_y + threadIdx.y; y_abs_offset < N && y_abs_offset < slide_y + data_block_size; y_abs_offset += thread_block_size) {
                sub_B[OFFSET(data_block_size, x_abs_offset - s, y_abs_offset - slide_y)] = B[OFFSET(N, x_abs_offset, y_abs_offset)];
            }
        }

        __syncthreads();

        for (int output_x = slide_x + threadIdx.x, i_x = 0; output_x < M && output_x < slide_x + data_block_size; output_x += thread_block_size, ++i_x) {
            for (int output_y = slide_y + threadIdx.y, i_y = 0; output_y < N && output_y < slide_y + data_block_size; output_y += thread_block_size, ++i_y) {
                for (int k = 0; k < data_block_size && s + k < K; ++k) {
                    tmp_res[i_x][i_y] += sub_A[OFFSET(data_block_size, output_x - slide_x, k)] * sub_B[OFFSET(data_block_size, k, output_y - slide_y)];
                }
            }
        }

        __syncthreads();
    }

    for (int output_x = slide_x + threadIdx.x, i_x = 0; output_x < M && output_x < slide_x + data_block_size; output_x += thread_block_size, ++i_x) {
        for (int output_y = slide_y + threadIdx.y, i_y = 0; output_y < N && output_y < slide_y + data_block_size; output_y += thread_block_size, ++i_y) {
            C[OFFSET(N, output_x, output_y)] = tmp_res[i_x][i_y];
        }
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
    constexpr size_t THREAD_SCALE = 2;
    dim3 blk(BLOCK_SIZE / THREAD_SCALE, BLOCK_SIZE / THREAD_SCALE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    constexpr size_t smem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    sgemm<THREAD_SCALE><<<grid, blk, smem_size>>>(M, N, K, A, B, C);
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