#!/bin/bash

cd ..
python3 start.py -bs 1000 -w 4 -ns 25000 -pa "outputs/dataset_exp" -s "adnexus" -d "random"
python3 start.py -bs 1000 -w 4 -ns 25000 -pa "outputs/dataset_exp" -s "adnexus" -d "wikitext"
python3 start.py -bs 1000 -w 4 -ns 25000 -pa "outputs/dataset_exp" -s "adnexus" -d "bookcorpus"
python3 start.py -bs 1000 -w 4 -ns 25000 -pa "outputs/dataset_exp" -s "adnexus" -d "wmt19"
python3 start.py -bs 1000 -w 4 -ns 25000 -pa "outputs/dataset_exp" -s "adnexus" -d "sst2"