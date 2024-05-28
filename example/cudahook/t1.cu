#include <stdio.h>

int main(){

  int a, *d_a;
  cudaMalloc(&d_a, sizeof(d_a[0]));
  cudaMemcpy(d_a, &a, sizeof(a), cudaMemcpyHostToDevice);
  cudaStream_t str;
  cudaStreamCreate(&str);
  cudaMemcpyAsync(d_a, &a, sizeof(a), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_a, &a, sizeof(a), cudaMemcpyHostToDevice, str);
  cudaDeviceSynchronize();
}