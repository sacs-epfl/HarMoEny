#!/bin/bash

python3 collect_nvidia_stats.py &
STATS_PID=$!

python3 start.py 8 12345 naive

kill -SIGTERM $STATS_PID

wait $STATS_PID