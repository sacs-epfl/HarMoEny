#!/bin/bash

nsys profile --trace=cuda,osrt,nvtx --force-overwrite=true --output=cuda_trace bash run_harmony.sh