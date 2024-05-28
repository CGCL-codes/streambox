#include "pool.h"

StreamPool::StreamPool(int initial_size) {
    for(int i = 0; i < initial_size; i++) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        available_streams.push(stream);
    }
}

StreamPool::~StreamPool() {
    while(!available_streams.empty()) {
        cudaStream_t stream = available_streams.front();
        available_streams.pop();
        cudaStreamDestroy(stream);
    }
}

cudaStream_t StreamPool::getStream() {
    std::lock_guard<std::mutex> lock(mutex);
    if (available_streams.empty()) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        available_streams.push(stream);
    }
    cudaStream_t stream = available_streams.front();
    available_streams.pop();
    return stream;
}

void StreamPool::returnStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex);
    available_streams.push(stream);
}
