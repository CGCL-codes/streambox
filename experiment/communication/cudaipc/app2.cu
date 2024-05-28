#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define DSIZE (1024 * 1024 * 16)

void checkCudaErrors(cudaError_t err, const char* msg, const char* file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(err), file, line);
        fprintf(stderr, "*** FAILED - ABORTING\n");
        exit(1);
    }
}

#define cudaCheckErrors(msg) checkCudaErrors(cudaGetLastError(), msg, __FILE__, __LINE__)

__global__ void test_kernel(float *data, float val) {
    data[threadIdx.x] = val;
}

__global__ void warmup(float *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < DSIZE) {
        data[idx] = (float)idx;
    }
}

void warmupDevice() {
    float *w_data;
    cudaMalloc(&w_data, 100*sizeof(float));
    for (size_t i = 0; i < 50; i++) {
        warmup<<<1, 100>>>(w_data);
    }
    cudaDeviceSynchronize();
    cudaFree(w_data);
}

void handleIPCRead(unsigned char* handle_buffer1, unsigned char* handle_buffer2) {
    FILE *fp;
    while (access("testfifo1", F_OK) == -1);
    fp = fopen("testfifo1", "r");
    if (!fp) {
        printf("fifo1 open fail \n");
        exit(1);
    }

    for (int i = 0; i < sizeof(cudaIpcMemHandle_t); i++) {
        int ret = fscanf(fp,"%c", &(handle_buffer1[i]));
        if (ret != 1) printf("ret = %d\n", ret);
    }
    for (int i = 0; i < sizeof(cudaIpcMemHandle_t); i++) {
        int ret = fscanf(fp,"%c", &(handle_buffer2[i]));
        if (ret != 1) printf("ret = %d\n", ret);
    }
    fclose(fp);
}

void executeKernelAndPrintTiming(float* data, float value, const char* message) {
    cudaEvent_t e_start, e_stop;
    float milliseconds = 0;

    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start);
    test_kernel<<<1, THREADS_PER_BLOCK>>>(data, value);
    cudaEventRecord(e_stop);
    cudaEventSynchronize(e_stop);
    cudaCheckErrors("kernel fail");

    cudaEventElapsedTime(&milliseconds, e_start, e_stop);
    printf(message, milliseconds * 1000);
}

int main() {
    cudaSetDevice(0);

    warmupDevice();

    unsigned char handle_buffer1[sizeof(cudaIpcMemHandle_t) + 1], handle_buffer2[sizeof(cudaIpcMemHandle_t) + 1];
    memset(handle_buffer1, 0, sizeof(cudaIpcMemHandle_t) + 1);
    memset(handle_buffer2, 0, sizeof(cudaIpcMemHandle_t) + 1);

    handleIPCRead(handle_buffer1, handle_buffer2);

    cudaIpcMemHandle_t my_handle1, my_handle2;
    memcpy((unsigned char *)(&my_handle1), handle_buffer1, sizeof(my_handle1));
    memcpy((unsigned char *)(&my_handle2), handle_buffer2, sizeof(my_handle2));

    float *other_g_data1, *other_g_data2;

    cudaIpcOpenMemHandle((void**)&other_g_data1, my_handle1, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("open IPC handle1 fail");

    cudaIpcOpenMemHandle((void**)&other_g_data2, my_handle2, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("open IPC handle2 fail");

    executeKernelAndPrintTiming(other_g_data1, 2.0f, "NO1 kernel + handle1 time = %f us\n");
    executeKernelAndPrintTiming(other_g_data1, 3.0f, "NO2 kernel + handle1 time = %f us\n");
    executeKernelAndPrintTiming(other_g_data1, 5.0f, "NO3 kernel + handle1 time = %f us\n");
    executeKernelAndPrintTiming(other_g_data2, 2.0f, "NO1 kernel + handle2 time = %f us\n");
    executeKernelAndPrintTiming(other_g_data2, 3.0f, "NO2 kernel + handle2 time = %f us\n");
    executeKernelAndPrintTiming(other_g_data2, 5.0f, "NO3 kernel + handle2 time = %f us\n");

    cudaIpcCloseMemHandle(other_g_data1);
    cudaIpcCloseMemHandle(other_g_data2);
    cudaCheckErrors("close IPC handle fail");

    FILE *fp = fopen("testfifo2", "w");
    if (!fp) {
        printf("fifo2 open fail \n");
        exit(1);
    }
    fprintf(fp, "1");
    fclose(fp);

    return 0;
}
