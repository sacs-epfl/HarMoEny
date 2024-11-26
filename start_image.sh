# https://hub.docker.com/r/gurobi/python for licensing

docker run -it \
    --gpus all \
    --ipc=host \
    -v .:/workspace \
    -v /raid/citadel/cache:/cache \
    -v ~/licenses/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
    moe:latest \
    /bin/bash