#!/bin/bash

max=0
for i in $(seq 1 10);
do
    result=$(python3 finding_equivalent_tokens.py)

    if (( result > max )); then
        max=$result
    fi
done

echo "Max eq tokens from 10 runs: $max"