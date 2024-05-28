#!/bin/bash

# The number of times to run the Python script. Can be overridden by command line argument.
runs="${1:-50}"

# Clear old result files
> context_init_times.txt
> tensor_alloc_times.txt
> model_load_times.txt

# Run the Python program the specified number of times
for ((i=1; i<=runs; i++))
do
    echo "Running iteration $i/$runs..."
    python tf.py | while read line 
    do
        # Classify output into different files by type
        if [[ $line == 'Context Initialization Time:'* ]]; then
            echo ${line#*:} >> context_init_times.txt
        elif [[ $line == 'Tensor1 Allocation Time:'* ]]; then
            echo ${line#*:} >> tensor_alloc_times.txt
        elif [[ $line == 'Model Load Time:'* ]]; then
            echo ${line#*:} >> model_load_times.txt
        fi
    done
    if [[ $? -ne 0 ]]; then
        echo "Python script failed on iteration $i. Exiting..."
        exit 1
    fi
done
