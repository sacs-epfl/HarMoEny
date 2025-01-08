#!/bin/bash

# $! is the script to profile

nsys profile --trace=cuda,osrt,nvtx --force-overwrite=true --output=cuda_trace bash $1