#include "../utils/cuda_context.cuh"

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

template<unsigned int NUM_PER_BLOCK, unsigned int WARPS_PER_BLOCK>
__global__ void reduce(float* d_in, float* d_out) {
    extern __shared__ volatile float shm[];

    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    float* d_in_begin = d_in + NUM_PER_BLOCK * block_id;

    float sum = 0;
    int NUM_PER_THREAD = (NUM_PER_BLOCK + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        sum += d_in_begin[thread_id + i * THREAD_PER_BLOCK];
    }

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1); 

    const int WARP_ID = thread_id / WARP_SIZE;
    const int LANE_ID = thread_id % WARP_SIZE;

    if (LANE_ID == 0) {
        shm[WARP_ID] = sum;
    }
    __syncthreads();

    sum = (LANE_ID < WARPS_PER_BLOCK) ? shm[LANE_ID] : 0;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1); 

    if (thread_id == 0) {
        d_out[block_id] = sum;
    }
}

int main() {
    constexpr int N = 32 * 1024 * 1024;
    float* h_in = (float*)malloc(N * sizeof(float));
    float* d_in;
    constexpr int NUM_BLOCKS = 1024;
    constexpr int NUM_PER_BLOCK = N / NUM_BLOCKS;
    float* h_out = (float*)malloc(NUM_BLOCKS * sizeof(float));
    float* std_out = (float*)malloc(NUM_BLOCKS * sizeof(float));
    float* d_out;

    {
        CudaContext cctxt;
        cudaError_t res = cctxt.alloc_resources((void**)&d_in, N * sizeof(float));
        if (res != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(res));
            return -1;
        }
        res = cctxt.alloc_resources((void**)&d_out, NUM_BLOCKS * sizeof(float));
        if (res != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(res));
            return -1;
        }

        // prepare datas
        for (int i = 0; i < N; ++i) {
            h_in[i] = 2;
        }

        for (int i = 0; i < NUM_BLOCKS; ++i) {
            float tmp = 0;
            for (int j = 0; j < NUM_PER_BLOCK; ++j) {
                tmp += h_in[i * NUM_PER_BLOCK + j];
            }
            std_out[i] = tmp;
        }

        // for (int i = 0; i < NUM_BLOCKS; ++i) {
        //     if (std_out[i] != 1 * THREAD_PER_BLOCK) {
        //         printf("check stdout array failed\n");
        //         break;
        //     }
        // }

        // copy h_in to d_in
        res = cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
        if (res != cudaSuccess) {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(res)); 
            return -1;
        }

        dim3 grid(NUM_BLOCKS, 1);
        dim3 blk(THREAD_PER_BLOCK, 1);
        int shared_mem_size = WARP_SIZE * sizeof(float);
        constexpr int warps_per_block = THREAD_PER_BLOCK / WARP_SIZE;
        reduce<NUM_PER_BLOCK, warps_per_block><<<grid, blk, shared_mem_size>>>(d_in, d_out);

        res = cudaMemcpy(h_out, d_out, sizeof(float) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
        if (res != cudaSuccess) {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(res)); 
            return -1;
        }

        for (int i = 0; i < NUM_BLOCKS; ++i) {
            if (h_out[i] != std_out[i]) {
                printf("check host result failed: %d: %lf expected, get %lf\n", i, std_out[i], h_out[i]);
                break;
            }
        }
    }

    free(std_out);
    free(h_out);
    free(h_in);
    return 0;
}