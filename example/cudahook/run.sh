#!/bin/bash
CURRENT_DIR=$(pwd)

g++ -I/usr/local/cuda/include -fPIC -shared -o libmylib.so mylib.cpp -ldl -L/usr/local/cuda/lib64 -lcudart

nvcc -o t1 t1.cu -cudart shared

LD_LIBRARY_PATH=/usr/local/cuda/lib64 LD_PRELOAD=./libmylib.so cuda-memcheck ./t1