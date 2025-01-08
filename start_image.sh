# https://hub.docker.com/r/gurobi/python for licensing

CACHE_PATH=$(realpath ../cache)

echo "Using cache located at $CACHE_PATH"

docker run -it \
    --gpus all \
    --ipc=host \
    -v .:/workspace \
    -v $CACHE_PATH:/cache \
    -v ./licenses/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
    -e HF_HOME=/cache \
    moe:latest \
    /bin/bash
