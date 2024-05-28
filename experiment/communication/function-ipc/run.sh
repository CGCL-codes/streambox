# Experiment: function ipc
# step 0: set env
device_id=3 # set your device here
export CUDA_VISIBLE_DEVICES=$device_id 
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# step 1: build the docker image
docker build --no-cache -t torch:streambox -f ../docker/Dockerfile.torch .

# step 2: run the container
docker run --name torch-commu-a -e CUDA_VISIBLE_DEVICES=${device_id} --gpus all --runtime nvidia -ti -d torch:streambox python func-b.py
docker run --name torch-commu-b -e CUDA_VISIBLE_DEVICES=${device_id} --gpus all --runtime nvidia -ti -d torch:streambox python func-a.py

# step 3: exec the container
# docker exec -ti torch-start /bin/bash
docker wait torch-commu-a
docker wait torch-commu-b
docker logs torch-commu-a
docker logs torch-commu-b