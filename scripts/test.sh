#!/bin/bash

cd ..
python3 start.py -w 4 -me 2 -bs 1000 -ni 10 -pa "outputs/new_run" -s "drop" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ni 10 -pa "outputs/new_run" -s "deepspeed" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ni 10 -pa "outputs/new_run" -s "adnexus" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ni 10 -pa "outputs/new_run" -s "adfabricus" -d "bookcorpus"
