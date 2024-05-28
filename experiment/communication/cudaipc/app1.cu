#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>

#define THREADS_PER_BLOCK 256
#define DSIZE (1024 * 1024 * 64)

// Error checking utility for CUDA calls
void checkCudaErrors(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",
                msg, cudaGetErrorString(err),
                __FILE__, __LINE__);
        exit(1);
    }
}

// Utility to measure elapsed time in microseconds
double getElapsedTime(const timespec &start, const timespec &stop) {
    return (stop.tv_sec - start.tv_sec) * 1e6 + 
           (stop.tv_nsec - start.tv_nsec) / 1e3;
}

void getIpcHandle(cudaIpcMemHandle_t *handle, float *data) {
    timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    checkCudaErrors(cudaIpcGetMemHandle(handle, data), "get IPC handle fail");
    clock_gettime(CLOCK_MONOTONIC, &stop);
    double elapsedTime = getElapsedTime(start, stop);
    printf("get IPC handle time = %f us\n", elapsedTime);
}

void checkResult(float *result, int length) {
    for (int i = 0; i < length; i++) {
        if (result[i] != 5.0f) {
            printf("result[%d] = %f\n", i, result[i]);
        }
    }
}

int main() {
    // Init device
    checkCudaErrors(cudaSetDevice(0), "Failed to set CUDA device");

    // Ensure FIFOs do not exist before starting
    unlink("testfifo1");
    unlink("testfifo2");

    if (mkfifo("testfifo1", 0600) != 0 || mkfifo("testfifo2", 0600) != 0) {
        perror("mkfifo error");
        return 1;
    }

    float *data1, *data2;
    checkCudaErrors(cudaMalloc(&data1, DSIZE * sizeof(float)), "malloc fail");
    checkCudaErrors(cudaMalloc(&data2, DSIZE * sizeof(float)), "malloc fail");

    checkCudaErrors(cudaMemset(data1, 0, DSIZE * sizeof(float)), "memset fail");
    checkCudaErrors(cudaMemset(data2, 0, DSIZE * sizeof(float)), "memset fail");

    cudaIpcMemHandle_t handle1, handle2;
    getIpcHandle(&handle1, data1);
    getIpcHandle(&handle2, data2);

    unsigned char handleBuffer1[sizeof(handle1)], handleBuffer2[sizeof(handle2)];
    memcpy(handleBuffer1, &handle1, sizeof(handle1));
    memcpy(handleBuffer2, &handle2, sizeof(handle2));

    timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);

    FILE *fp;
    printf("waiting for app2\n");
    fp = fopen("testfifo1", "w");
    if (!fp) {
        perror("fifo1 open fail");
        return 1;
    }
    fwrite(handleBuffer1, 1, sizeof(handle1), fp);
    fwrite(handleBuffer2, 1, sizeof(handle2), fp);
    fclose(fp);

    // Wait for app2's notification
    char notify;
    fp = fopen("testfifo2", "r");
    if (!fp) {
        perror("fifo2 open fail");
        return 1;
    }
    fscanf(fp, "%c", &notify);
    fclose(fp);

    // Stop the timer and calculate the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &stop);
    double elapsedTime = getElapsedTime(start, stop);
    printf("end-to-end time = %f us\n", elapsedTime);

    float *result = (float *)malloc(DSIZE * sizeof(float));
    checkCudaErrors(cudaMemcpy(result, data1, DSIZE * sizeof(float), cudaMemcpyDeviceToHost), "memcpy fail");
    checkResult(result, THREADS_PER_BLOCK);

    checkCudaErrors(cudaMemcpy(result, data2, DSIZE * sizeof(float), cudaMemcpyDeviceToHost), "memcpy fail");
    checkResult(result, THREADS_PER_BLOCK);

    printf("data1 size = %f MB\n", DSIZE * sizeof(float) / 1000000.0f);

    free(result);
    unlink("testfifo1");
    unlink("testfifo2");
    return 0;
}
