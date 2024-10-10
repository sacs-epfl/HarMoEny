#!/bin/bash

cd ..
python3 start.py -w 1 -me 8 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "drop" -d "bookcorpus"
python3 start.py -w 2 -me 4 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "drop" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "drop" -d "bookcorpus"
python3 start.py -w 8 -me 1 -bs 800 -ns 64000 -pa "outputs/num_gpus_exp" -s "drop" -d "bookcorpus"

python3 start.py -w 1 -me 8 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -w 2 -me 4 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -w 8 -me 1 -bs 800 -ns 64000 -pa "outputs/num_gpus_exp" -s "deepspeed" -d "bookcorpus"

python3 start.py -w 1 -me 8 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -w 2 -me 4 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -w 8 -me 1 -bs 800 -ns 64000 -pa "outputs/num_gpus_exp" -s "adnexus" -d "bookcorpus"

python3 start.py -w 1 -me 8 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "demeter" -d "bookcorpus"
python3 start.py -w 2 -me 4 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "demeter" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "demeter" -d "bookcorpus"
python3 start.py -w 8 -me 1 -bs 800 -ns 64000 -pa "outputs/num_gpus_exp" -s "demeter" -d "bookcorpus"

python3 start.py -w 1 -me 8 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "even_split" -d "bookcorpus"
python3 start.py -w 2 -me 4 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "even_split" -d "bookcorpus"
python3 start.py -w 4 -me 2 -bs 1000 -ns 64000 -pa "outputs/num_gpus_exp" -s "even_split" -d "bookcorpus"
python3 start.py -w 8 -me 1 -bs 800 -ns 64000 -pa "outputs/num_gpus_exp" -s "even_split" -d "bookcorpus"

