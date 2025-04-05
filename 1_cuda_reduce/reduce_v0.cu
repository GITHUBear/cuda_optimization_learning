#include "../utils/cuda_context.cuh"

#define THREAD_PER_BLOCK 256

__global__ void reduce(float* d_in, float* d_out) { 
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int num_threads_per_block = blockDim.x;
    float* d_in_begin = d_in + num_threads_per_block * block_id;

#pragma unroll
    for (int i = 1; i < num_threads_per_block; i <<= 1) {
        if (thread_id % (i << 1) == 0) {
            d_in_begin[thread_id] += d_in_begin[thread_id + i];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        d_out[block_id] = d_in_begin[0];
    }
}

int main() {
    constexpr int N = 32 * 1024 * 1024;
    float* h_in = (float*)malloc(N * sizeof(float));
    float* d_in;
    constexpr int NUM_BLOCKS = N / THREAD_PER_BLOCK;
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
            for (int j = 0; j < THREAD_PER_BLOCK; ++j) {
                tmp += h_in[i * THREAD_PER_BLOCK + j];
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
        reduce<<<grid, blk>>>(d_in, d_out);

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