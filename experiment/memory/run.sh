# Experiment: memory footprint in MPS
# step 0: set env
device_id=3 # set your device here
export CUDA_VISIBLE_DEVICES=$device_id 
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# step 1: run the following command to set specific GPU to exclusive mode
# nvidia-smi -L # List the GPU's on node.
nvidia-smi -q | grep "Compute Mode" # List GPU state and configuration information.
nvidia-smi -i $device_id -c EXCLUSIVE_PROCESS # Set GPU 0 to exclusive mode, run as root.
nvidia-smi -q | grep "Compute Mode" # check whether the GPU is in exclusive mode

# step 2: run the following command to start and set MPS
nvidia-cuda-mps-control -d
echo set_default_active_thread_percentage 10 | nvidia-cuda-mps-control
# ps -ef | grep mps

# step 3: build the docker image
docker build --no-cache -t torch:streambox -f ../docker/Dockerfile.torch .

# step 4: run the container
for ((i=0;i<$1;i++))
do
    docker run --name torch-mps-${i} -e CUDA_VISIBLE_DEVICES=${device_id} --gpus all --runtime nvidia -d torch:streambox python app.py # docker run --name torch-mps -e CUDA_VISIBLE_DEVICES=3 --gpus all --runtime nvidia -d torch:mps python app.py
done
docker ps
for ((i=0;i<$1;i++))
do
    docker wait torch-mps-${i}
    docker logs torch-mps-${i}
    docker rm torch-mps-${i}
done
# step 5: stop and unset MPS
echo quit | nvidia-cuda-mps-control
nvidia-smi -i $device_id -c DEFAULT # run as root
