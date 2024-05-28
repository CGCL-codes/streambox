#ifndef STREAM_BOX
#define STREAM_BOX
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

#include <time.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
void init_float(float* ip,int size)
{
  time_t t;
  srand((unsigned )time(&t));
  for(int i=0;i<size;i++)
  {
    ip[i]=(float)(rand()&0xffff)/1000.0f;
  }
}

void init_int(int* ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i<size; i++)
	{
		ip[i] = int(rand()&0xff);
	}
}

void init_0int(int* ip, int size)
{
	for (int i = 0; i<size; i++)
	{
		ip[i] = 0;
	}
}

void init_0float(float* ip,int size)
{
  for(int i = 0; i < size; i++)
  {
    ip[i] = (float)0;
  }
}

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("Using device %d: %s\n",dev,deviceProp.name);
  CHECK(cudaSetDevice(dev));

}
#endif