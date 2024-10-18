#!/bin/bash

cd ..
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "drop" -d "random"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "drop" -d "wikitext"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "drop" -d "bookcorpus"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "drop" -d "wmt19"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "drop" -d "sst2"

python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "deepspeed" -d "random"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "deepspeed" -d "wikitext"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "deepspeed" -d "bookcorpus"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "deepspeed" -d "wmt19"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "deepspeed" -d "sst2"

python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adnexus" -d "random"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adnexus" -d "wikitext"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adnexus" -d "wmt19"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adnexus" -d "sst2"

python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "demeter" -d "random"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "demeter" -d "wikitext"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "demeter" -d "bookcorpus"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "demeter" -d "wmt19"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "demeter" -d "sst2"

python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "even_split" -d "random"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "even_split" -d "wikitext"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "even_split" -d "bookcorpus"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "even_split" -d "wmt19"
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "even_split" -d "sst2"

python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "random" -cp RAND
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "wikitext" -cp RAND
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "bookcorpus" -cp RAND
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "wmt19" -cp RAND
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "sst2" -cp RAND

python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "random" -cp MTU
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "wikitext" -cp MTU
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "bookcorpus" -cp MTU
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "wmt19" -cp MTU
python3 start.py -bs 1000 -w 4 -ni 20 -ec 2 -pa "outputs/dataset_exp" -s "adfabricus" -d "sst2" -cp MTU