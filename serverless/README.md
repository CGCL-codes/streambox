# Serverless

## Description

The source code of the required serverless services in StreamBox.

## Component

### Load Generator
It is responsible for simulating and generating the workload according to the specific patterns, then sending these generated requests to the Batcher service for further processing. It helps to mimic the real-world scenarios where requests arrive in batches at the server.

The generation of requests is influenced by several parameters that can be adjusted to fit specific workload characteristics:

* `multipile`: a factor that scales the total number of requests to generate.
* `len`: the length of the workload (in time units).
* `gap_time`: the time interval between two consecutive bursts of requests.
* `lap_time`: the duration of the idle period between two bursts.
* `type`: the pattern of the workload ('bursty', 'periodic', 'sporadic', etc.)

The Load Generator reads these parameters and the workload pattern from a specified file and generates the corresponding workload. It then sends these requests to the Batcher service via HTTP.

### Batcher
Service that provides the batch processing of the requests and send the batched requests to the inference server. It has two important hyperparameters:

* Model-allowed Maximum Batch Size (i.e., the inference server will schedule the batched input once a certain number of inputs are collected)
* Batching Timewindow (i.e., the longest time period the inference server will wait for inputs to form a batch)

And it support multi-tenancy, which means it can serve multiple requests from different users at the same time, with NVIDIA MPS as isolation.

## Contents

```txt
> tree ./
./
├── READEME.md
├── config
│   └── test.yaml
├── go.mod
├── go.sum
├── pkg
│   ├── config
│   │   └── config.go
│   ├── domain
│   │   ├── batcher.go
│   │   ├── data.go
│   │   └── req.go
│   ├── env
│   │   └── env.go
│   ├── log
│   │   └── log.go
│   ├── statuserror
│   │   ├── error.go
│   │   ├── error_test.go
│   │   └── gin.go
│   ├── util
│   │   ├── const.go
│   │   └── util.go
│   └── x
│       └── gin.go
├── service
│   └── batcher
│       ├── config.go
│       ├── handler.go
│       ├── impl
│       │   ├── batcher.go
│       │   ├── data.go
│       │   └── req.go
│       └── main
│           └── main.go
└── workload
    ├── __pycache__
    │   └── config.cpython-310.pyc
    ├── bursty-load.txt
    ├── config.py
    ├── gen_workload.py
    ├── load_generator.py
    ├── periodic-load.txt
    ├── requirements.txt
    └── sporadic-load.txt
```

## Usage
### Load Generator
1. Download the source code

    ```bash
    git clone https://github.com/streambox2024/streambox
    ```
2. Download and Configure [Python3](https://www.python.org/downloads/)
3. Set the environment variables

    ```bash
    export STREAMBOX_ROOT=PATH_TO_STREAMBOX
    export CONFIG_FILE=$STREAMBOX_ROOT/serverless/config/test.yaml
    ```
4. To generate and send requests, run the load_generator.py script:

    ```shell
    cd $STREAMBOX_ROOT/serverless/workload
    pip3 install -r requirements.txt
    python3 load_generator.py
    ```    
### Batcher
1. Download the source code

    ```bash
    git clone https://github.com/streambox2024/streambox
    ```
2. Download and Configure [Golang](https://go.dev/doc/install)
3. Set the environment variables

    ```bash
    export STREAMBOX_ROOT=PATH_TO_STREAMBOX
    export CONFIG_FILE=$STREAMBOX_ROOT/serverless/config/test.yaml
    ```
4. Run the service

    Take `batcher` as an example:
    ```bash
    cd $STREAMBOX_ROOT/serverless
    go mod tidy
    go run ./service/batcher/main/main.go
    ```
## Implement your services
### Batcher
Implement your own batcher by using the `Batcher` interface in `./pkg/domain/batcher.go`
```go
type Batcher interface {
	SetTimeWindow(timeWindow int)
	SetBatchSize(batchSize int)
	AddOne(ctx context.Context, req Request) error
	Batch(ctx context.Context, model string, reason string) (interface{}, error)
}
```

## Performance