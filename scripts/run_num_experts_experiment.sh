#!/bin/bash

# TODO need to figure out the max batch sizes

cd ..
python3 start.py -e 8 -me 2 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "drop" -d "bookcorpus"
python3 start.py -e 16 -me 4 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "drop" -d "bookcorpus"
python3 start.py -e 32 -me 8 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "drop" -d "bookcorpus"
python3 start.py -e 64 -me 16 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "drop" -d "bookcorpus"
python3 start.py -e 128 -me 32 -bs 800 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "drop" -d "bookcorpus"

python3 start.py -e 8 -me 2 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -e 16 -me 4 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -e 32 -me 8 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -e 64 -me 16 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -e 128 -me 32 -bs 800 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "deepspeed" -d "bookcorpus"

python3 start.py -e 8 -me 2 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -e 16 -me 4 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -e 32 -me 8 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -e 64 -me 16 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -e 128 -me 32 -bs 800 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "adnexus" -d "bookcorpus"

python3 start.py -e 8 -me 2 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "demeter" -d "bookcorpus"
python3 start.py -e 16 -me 4 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "demeter" -d "bookcorpus"
python3 start.py -e 32 -me 8 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "demeter" -d "bookcorpus"
python3 start.py -e 64 -me 16 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "demeter" -d "bookcorpus"
python3 start.py -e 128 -me 32 -bs 800 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "demeter" -d "bookcorpus"

python3 start.py -e 8 -me 2 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "even_split" -d "bookcorpus"
python3 start.py -e 16 -me 4 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "even_split" -d "bookcorpus"
python3 start.py -e 32 -me 8 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "even_split" -d "bookcorpus"
python3 start.py -e 64 -me 16 -bs 1000 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "even_split" -d "bookcorpus"
python3 start.py -e 128 -me 32 -bs 800 -w 4 -ns 25000 -pa "outputs/num_experts_exp" -s "even_split" -d "bookcorpus"