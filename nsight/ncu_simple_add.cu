#include <cstdio>
#include "../utils/cuda_context.cuh"

#define THREAD_PER_BLOCK 256

__global__ void add(int* a, int* b, int* c, int N) {
    int idx = blockIdx.x * THREAD_PER_BLOCK + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    constexpr int num_eles = 32 * 1024 * 1024;
    int* h_a = new int[num_eles];
    for (int i = 0; i < num_eles; ++i) h_a[i] = 1;
    int* h_b = new int[num_eles];
    for (int i = 0; i < num_eles; ++i) h_b[i] = 2;
    int* h_c = new int[num_eles];
    for (int i = 0; i < num_eles; ++i) h_c[i] = 0;

    int* d_a, *d_b, *d_c;
    CudaContext cctxt;
    cctxt.alloc_resources((void**)&d_a, num_eles * sizeof(int));
    cctxt.alloc_resources((void**)&d_b, num_eles * sizeof(int));
    cctxt.alloc_resources((void**)&d_c, num_eles * sizeof(int));
    
    cudaMemcpy(d_a, h_a, num_eles * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_eles * sizeof(int), cudaMemcpyHostToDevice);

    add<<<num_eles / THREAD_PER_BLOCK,  THREAD_PER_BLOCK>>>(d_a, d_b, d_c, num_eles);

    cudaMemcpy(h_c, d_c, num_eles * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_eles; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("check failed\n");
            break;
        }
    }

    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);

    delete[] h_c;
    delete[] h_b;
    delete[] h_a;
    return 0;
}