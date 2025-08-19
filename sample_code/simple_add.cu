#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simple_add(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 256;
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
        h_c[i] = 0;   // Initialize host result array
    }

    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, N * sizeof(int));  // Initialize device result array

    // <<<blocks, threads>>> : one block of 256 threads
    simple_add<<<1, 256>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize(); // Wait for kernel to finish

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
