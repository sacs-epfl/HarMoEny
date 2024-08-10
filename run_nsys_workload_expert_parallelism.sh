#!/bin/bash

read -p "Choose filename (naive, bmm, demeter, etc): " file_name

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --force-overwrite=true --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o my_profile python3 run_workload_expert_parallelism.py $file_name