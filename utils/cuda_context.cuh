#include <vector>
#include <cstdio>

class CudaContext {
private:
    std::vector<void*> resources;
public:
    ~CudaContext() {
        for (int i = resources.size() - 1; i >= 0; --i) {
            cudaFree(resources[i]);
        }
        printf("CudaContext destruction\n");
    }

    cudaError_t alloc_resources(void** applicant, size_t size) {
        cudaError_t res = cudaMalloc(applicant, size);
        if (res == cudaSuccess) {
            resources.push_back(*applicant);
        }
        return res;
    }
};