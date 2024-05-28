#ifndef STREAM_POOL_H
#define STREAM_POOL_H

#include "cuda_runtime.h"
#include <mutex>
#include <queue>

class StreamPool {
private:
    std::queue<cudaStream_t> available_streams;
    std::mutex mutex;
public:
    StreamPool(int initial_size = 10);
    ~StreamPool();

    cudaStream_t getStream();
    void returnStream(cudaStream_t stream);
};

#endif //STREAM_POOL_H
