#include <cuda_runtime.h>
#include <iostream>
#include <atomic>

// Error checking macro and function
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void addKernel(int *a, int *b, int *c, int N, int t) {
    if (t == 1) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) {
            c[idx] = a[idx] + b[idx];
        }
    } else {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

__global__ void syncKernel(int *sync) {
    while (*((volatile int*)(sync)) != 1) {
        __threadfence_system();
    }
}

void logMemoryStatus(const char* message) {
    // Memory information variables
    size_t free_mem, total_mem;
    cudaCheck(cudaMemGetInfo(&free_mem, &total_mem));
    float free_gb = free_mem / (float)(1 << 30);  // Convert bytes to gigabytes
    float total_gb = total_mem / (float)(1 << 30);

    // Variables for graph memory attributes
    size_t usedMemCurrent, usedMemHigh, reservedMemCurrent, reservedMemHigh;

    // Retrieve graph memory usage information
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &usedMemCurrent));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &usedMemHigh));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemCurrent, &reservedMemCurrent));
    cudaCheck(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemHigh, &reservedMemHigh));

    // Print basic memory info
    std::cout << message << " - Free Memory: " << free_gb << " GB, Total Memory: " << total_gb << " GB, Graph Memory Usage: " << usedMemCurrent / (double)(1 << 30) << " GB, Graph Reserved Memory: " << reservedMemCurrent / (double)(1 << 30) << " GB\n";
}

struct HostFuncParams {
    int* c_ptr;
    int N;
    int* sync_data;
};

void host_func(HostFuncParams* params) {
    printf("============ host function called ============\n");
    for (int i = 0; i < params->N; ++i) {
        (params->c_ptr)[i] <<= 1;
    }
    asm volatile("mfence" ::: "memory");
    *((volatile int*)(params->sync_data)) = 1;
    printf("============ host function end ============\n");
}

int main() {
    cudaMemPool_t mempool;
    cudaDeviceGetDefaultMemPool(&mempool, 0);
    uint64_t threshold = 0; // UINT64_MAX;
    cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

    const int N = 1024 * 1024 * 256;
    const int bytes = N * sizeof(int);
    int *a, *b, *c, *h_c;

    int *sync_data;
    cudaCheck(cudaHostAlloc(&sync_data, sizeof(int), cudaHostAllocMapped));
    *sync_data = 0;

    int *d_sync_data;
    cudaCheck(cudaHostGetDevicePointer(&d_sync_data, sync_data, 0));

    // Allocate device memory for a and b
    cudaCheck(cudaMalloc(&a, bytes));
    cudaCheck(cudaMalloc(&b, bytes));
    cudaCheck(cudaMalloc(&c, bytes));

    // Initialize a and b on the host
    // int *h_a = new int[N];
    // int *h_b = new int[N];
    int *h_a, *h_b;
    cudaMallocHost(&h_a, N * sizeof(int));
    cudaMallocHost(&h_b, N * sizeof(int));
    cudaMallocHost(&h_c, N * sizeof(int));
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Copy data from host to device
    cudaCheck(cudaMemcpy(a, h_a, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b, h_b, bytes, cudaMemcpyHostToDevice));

    // Allocate host memory for the result
    // h_c = new int[N];

    // Create a stream
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));

    logMemoryStatus("before capture");

    HostFuncParams param;
    param.c_ptr = h_c;
    param.N = N;
    param.sync_data = sync_data;

    cudaEvent_t event;
    cudaEventCreate(&event);

    // Begin graph capture
    cudaGraph_t graph;
    cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Allocate memory for c during graph capture using cudaMallocAsync
    // cudaCheck(cudaMallocAsync(&c, bytes, stream));

    logMemoryStatus("inside capture, after cudaMallocAsync");

    // Launch the add kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    addKernel<<<grid, block, 0, stream>>>(a, b, c, N, 1);

    // Copy the output to CPU using cudaMemcpyAsync
    cudaCheck(cudaMemcpyAsync(h_c, c, bytes, cudaMemcpyDeviceToHost, stream));

    // host_func(&param);
    cudaLaunchHostFunc(stream, (cudaHostFn_t)host_func, (void*)(&param));
    // cudaCheck(cudaStreamSynchronize(stream));

    // cudaMemPrefetchAsync()

    // syncKernel<<<1,1,0,stream>>>(d_sync_data);

    cudaCheck(cudaMemcpyAsync(a, h_c, bytes, cudaMemcpyHostToDevice, stream));

    cudaEventRecord(event, stream);
    cudaStreamWaitEvent(stream, event);

    addKernel<<<grid, block, 0, stream>>>(a, b, c, N, 2);

    cudaCheck(cudaMemcpyAsync(h_c, c, bytes, cudaMemcpyDeviceToHost, stream));

    // cudaCheck(cudaStreamSynchronize(stream));

    // End graph capture
    cudaCheck(cudaStreamEndCapture(stream, &graph));

    // Launch the graph
    cudaGraphExec_t graphExec;
    cudaCheck(cudaGraphInstantiateWithFlags(&graphExec, graph));

    // logMemoryStatus("before execution");

    cudaCheck(cudaGraphLaunch(graphExec, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    // logMemoryStatus("after the first execution");

    // Check result
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != 2*(h_a[i] + h_b[i]) + h_b[i]) {
            correct = false;
            printf("%d: %d expected, %d get\n", i, 2*(h_a[i] + h_b[i])+h_b[i], h_c[i]);
            break;
        }
    }
    if (correct) {
        std::cout << "Results are correct!" << std::endl;
    } else {
        std::cout << "Results are incorrect!" << std::endl;
    }

    // Cleanup

    cudaCheck(cudaGraphDestroy(graph));
    cudaCheck(cudaGraphExecDestroy(graphExec));
    cudaCheck(cudaEventDestroy(event));

    cudaCheck(cudaFree(a));
    cudaCheck(cudaFree(b));
    cudaCheck(cudaFree(c));
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    // delete[] h_a;
    // delete[] h_b;
    // delete[] h_c;

    cudaCheck(cudaStreamDestroy(stream));

    return 0;
}