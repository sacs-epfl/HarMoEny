#!/bin/bash

nsys profile --trace=cuda,osrt,nvtx --force-overwrite=true --output=time_cuda_trace bash run_time_times.sh