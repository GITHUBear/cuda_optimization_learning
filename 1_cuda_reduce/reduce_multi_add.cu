#include "../utils/cuda_context.cuh"

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

__device__ void warpReduce(volatile float* shm, int thread_id) {
    shm[thread_id] += shm[thread_id + 32];
    shm[thread_id] += shm[thread_id + 16]; 
    shm[thread_id] += shm[thread_id + 8]; 
    shm[thread_id] += shm[thread_id + 4]; 
    shm[thread_id] += shm[thread_id + 2]; 
    shm[thread_id] += shm[thread_id + 1]; 
}

template<unsigned int NUM_PER_BLOCK>
__global__ void reduce(float* d_in, float* d_out) {
    extern __shared__ volatile float shm[];

    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    float* d_in_begin = d_in + NUM_PER_BLOCK * block_id;

    shm[thread_id] = 0;
    int NUM_PER_THREAD = (NUM_PER_BLOCK + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        // shm[thread_id] += d_in_begin[thread_id + i * THREAD_PER_BLOCK];
        shm[thread_id] += d_in_begin[thread_id * NUM_PER_THREAD + i];
    }
    
    // shm[thread_id] = d_in_begin[thread_id] + d_in_begin[thread_id + THREAD_PER_BLOCK];
    __syncthreads();

// #pragma unroll
//     for (int i = 0; (num_threads_per_block >> (1 + i)) > WARP_SIZE; ++i) {
//         if ((thread_id << (i + 1)) < num_threads_per_block) {
//             // no_bank_conflict
//             shm[thread_id] += shm[thread_id + (THREAD_PER_BLOCK >> (i + 1))];
//         }
//         __syncthreads();
//     }

    if (thread_id < 128)
        shm[thread_id] += shm[thread_id + 128];
    __syncthreads();
    if (thread_id < 64)
        shm[thread_id] += shm[thread_id + 64];
    __syncthreads();
// #pragma unroll
//     for (int i = 0; (1 << i) <= WARP_SIZE; ++i) {
//         shm[thread_id] += shm[thread_id + (WARP_SIZE >> i)];
//     }
    // If your implementation is the same as following, compiling with release mode will be fail in result checking.
    // Must declare shm with 'extern __shared__ volatile float shm[];' to disable disordering.
    if (thread_id < WARP_SIZE) {
        warpReduce(shm, thread_id);
    }
    // for (int i = 1; i < num_threads_per_block; i <<= 1) {
    //     if (thread_id * (i << 1) < num_threads_per_block) {
    //         shm[thread_id * (i << 1)] += shm[thread_id * (i << 1) + i];
    //     }
    //     __syncthreads();
    // }

    if (thread_id == 0) {
        d_out[block_id] = shm[0];
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
        int shared_mem_size = THREAD_PER_BLOCK * sizeof(float);
        reduce<NUM_PER_BLOCK><<<grid, blk, shared_mem_size>>>(d_in, d_out);

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