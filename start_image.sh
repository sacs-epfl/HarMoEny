docker run -it \
    --gpus all \
    --ipc=host \
    -v .:/workspace \
    -v /raid/citadel/cache:/cache \
    moe:latest \
    /workspace/docker_startup.sh