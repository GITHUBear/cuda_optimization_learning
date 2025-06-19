#include <cstdio>
#include <algorithm>
#include <iostream>

#define OFFSET(col, x, y) ((col) * (x) + (y))

void rand_matrix(int m, int n, float* M) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            M[OFFSET(n, i, j)] = 2.0 * (float)drand48() - 1.0;
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

__device__ void device_assert(bool condition, const char* msg, const char* file, int line) {
    if (!condition) {
        printf("Assertion failed: %s, file: %s, line: %d\n", msg, file, line);
        __threadfence();
        asm volatile("trap;");
    }
}

#define DEVICE_ASSERT(condition) \
    device_assert((condition), #condition, __FILE__, __LINE__)

__global__ void sgemm_v0(int M, int N, int K, float* A, float* B, float* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[OFFSET(K, x, i)] * B[OFFSET(N, i, y)];
        }
        C[OFFSET(N, x, y)] = tmp;
    }
}

template<
    const int BLOCK_TILE_M,
    const int BLOCK_TILE_K,
    const int BLOCK_TILE_N,
    const int THREAD_TILE_Y,
    const int THREAD_TILE_X
>
__global__ void sgemm(
    float* A,               // M * K
    float* B,               // K * N
    float* C,               // M * N
    int M, int K, int N,
    int stride_ay, int stride_ax,
    int stride_by, int stride_bx,
    int stride_cy, int stride_cx
) {
    int blk_x = blockIdx.x;
    int blk_y = blockIdx.y;
    int blk_dim_x = blockDim.x;
    int blk_dim_y = blockDim.y;
    int trd_x = threadIdx.x;
    int trd_y = threadIdx.y;
    int trd_id = trd_y * blk_dim_y + trd_x;
    int num_threads_per_blk = blk_dim_x * blk_dim_y;

    DEVICE_ASSERT(blk_dim_x * THREAD_TILE_X == BLOCK_TILE_N);
    DEVICE_ASSERT(blk_dim_y * THREAD_TILE_Y == BLOCK_TILE_M);
    DEVICE_ASSERT(BLOCK_TILE_K % 4 == 0);
    DEVICE_ASSERT(BLOCK_TILE_N % 4 == 0);

    __shared__ float sm_A[BLOCK_TILE_K][BLOCK_TILE_M];   // transposed
    __shared__ float sm_B[BLOCK_TILE_K][BLOCK_TILE_N];

    float reg_A_in[THREAD_TILE_Y];
    float reg_B_in[THREAD_TILE_X];
    float reg_C_out[THREAD_TILE_Y][THREAD_TILE_X];

    float* blk_a_ptr = A + blk_y * BLOCK_TILE_M * stride_ay;
    float* blk_b_ptr = B + blk_x * BLOCK_TILE_N * stride_bx;
    float* blk_c_ptr = C + blk_y * BLOCK_TILE_M * stride_cy + blk_x * BLOCK_TILE_N * stride_cx;

    for (int x = 0; x < THREAD_TILE_Y; ++x) {
        for (int y = 0; y < THREAD_TILE_X; ++y) {
            reg_C_out[x][y] = 0;
        }
    }
    for (int i = 0; i < K; i += BLOCK_TILE_K) {
        float* tile_blk_a_ptr = blk_a_ptr + i * stride_ax;
        float* tile_blk_b_ptr = blk_b_ptr + i * stride_by;
        // Load tile A to sm_A
        // move BLOCK_TILE_M * BLOCK_TILE_K floats
        // DO NOT CONSIDER CORNER CASE NOW
        DEVICE_ASSERT(min(BLOCK_TILE_K, K - i) == BLOCK_TILE_K);
        DEVICE_ASSERT(min(BLOCK_TILE_M, M - blk_y * BLOCK_TILE_M) == BLOCK_TILE_M);

        int total_tasks = (BLOCK_TILE_M * BLOCK_TILE_K) / 4;
        int row_tasks = BLOCK_TILE_K / 4;
        for (int j = trd_id; j < total_tasks; j += num_threads_per_blk) {
            int M_idx = j / row_tasks;
            int K_idx = (j % row_tasks) * 4;
            float4 ldg_f4 = 
                *(reinterpret_cast<float4*>(tile_blk_a_ptr + M_idx * stride_ay  + K_idx * stride_ax));
            sm_A[K_idx][M_idx] = ldg_f4.x;
            sm_A[K_idx + 1][M_idx] = ldg_f4.y;
            sm_A[K_idx + 2][M_idx] = ldg_f4.z;
            sm_A[K_idx + 3][M_idx] = ldg_f4.w;
        }

        // Load tile B to sm_B
        DEVICE_ASSERT(min(BLOCK_TILE_N, N - blk_x * BLOCK_TILE_N) == BLOCK_TILE_N);
        total_tasks = (BLOCK_TILE_K * BLOCK_TILE_N) / 4;
        row_tasks = BLOCK_TILE_N / 4;
        for (int j = trd_id; j < total_tasks; j += num_threads_per_blk) {
            int K_idx = j / row_tasks;
            int N_idx = (j % row_tasks) * 4;
            (reinterpret_cast<float4*>(&(sm_B[K_idx][N_idx])))[0] = 
                *(reinterpret_cast<float4*>(tile_blk_b_ptr + K_idx * stride_by + N_idx * stride_bx));
        }

        __syncthreads();

        for (int trd_tile_kidx = 0; trd_tile_kidx < BLOCK_TILE_K; ++trd_tile_kidx) {
            int trd_tile_midx = trd_y * THREAD_TILE_Y;
            int trd_tile_nidx = trd_x * THREAD_TILE_X;
            // Load tile sm_A to reg_A_in
            for (int j = 0; j < THREAD_TILE_Y; ++j) {
                reg_A_in[j] = sm_A[trd_tile_kidx][trd_tile_midx + j];
            }

            // Load tile sm_B to reg_B_in
            for (int j = 0; j < THREAD_TILE_X; ++j) {
                reg_B_in[j] = sm_B[trd_tile_kidx][trd_tile_nidx + j];
            }

            // calculate thread tile
            for (int reg_c_y = 0; reg_c_y < THREAD_TILE_Y; ++reg_c_y) {
                for (int reg_c_x = 0; reg_c_x < THREAD_TILE_X; ++reg_c_x) {
                    reg_C_out[reg_c_y][reg_c_x] += (reg_A_in[reg_c_y] * reg_B_in[reg_c_x]);
                }
            }
        }

        __syncthreads();
    }

    // Store reg_C_out to DRAM
    float* trd_tile_c_ptr = blk_c_ptr + trd_y * THREAD_TILE_Y * stride_cy + trd_x * THREAD_TILE_X * stride_cx;
    for (int reg_c_y = 0; reg_c_y < THREAD_TILE_Y; ++reg_c_y) {
        for (int reg_c_x = 0; reg_c_x < THREAD_TILE_X; ++reg_c_x) {
            *(trd_tile_c_ptr + reg_c_y * stride_cy + reg_c_x * stride_cx) = reg_C_out[reg_c_y][reg_c_x];
        }
    }
}

int main() {
    size_t M = 2048;
    size_t N = 2048;
    size_t K = 2048;

    float* A, *B, *C, *V0_C;
    cudaMallocManaged((void**)&A, M * K * sizeof(float));
    cudaMallocManaged((void**)&B, K * N * sizeof(float));
    cudaMallocManaged((void**)&C, M * N * sizeof(float));
    cudaMallocManaged((void**)&V0_C, M * N * sizeof(float));
    rand_matrix(M, K, A);
    rand_matrix(K, N, B);

    constexpr size_t BLOCK_TILE_M = 128;
    constexpr size_t BLOCK_TILE_N = 128;
    constexpr size_t BLOCK_TILE_K = 8;
    constexpr size_t THREAD_TILE_Y = 8;
    constexpr size_t THREAD_TILE_X = 8;

    dim3 blk_v0(16, 16);
    dim3 grid_v0(M / 16, N / 16);
    sgemm_v0<<<grid_v0, blk_v0>>>(M, N, K, A, B, V0_C);
    cudaDeviceSynchronize();
    std::cout << V0_C[0] << std::endl;

    dim3 blk(BLOCK_TILE_N / THREAD_TILE_X, BLOCK_TILE_M / THREAD_TILE_Y);
    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    sgemm<BLOCK_TILE_M, BLOCK_TILE_K, BLOCK_TILE_N, THREAD_TILE_Y, THREAD_TILE_X><<<grid, blk>>>(
        A, B, C,
        M, K, N,
        K, 1,
        N, 1,
        N, 1
    );
    cudaDeviceSynchronize();
    std::cout << C[0] << std::endl;

    float max_diff = compare_matrics(M, N, V0_C, C);
    if (max_diff > 0.01) {
        printf("result error!\n");
    } else {
        printf("check success!\n");
    }

    cudaFree(V0_C);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}