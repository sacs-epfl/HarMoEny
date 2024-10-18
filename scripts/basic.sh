#!/bin/bash

cd ..
python3 start.py -w 4 -ni 10 -bs 1000 -pa outputs/basic -d wikitext -e 8 -ec 2 -dm 768 -s drop
python3 start.py -w 4 -ni 10 -bs 1000 -pa outputs/basic -d wikitext -e 8 -ec 2 -dm 768 -s deepspeed
python3 start.py -w 4 -ni 10 -bs 1000 -pa outputs/basic -d wikitext -e 8 -ec 2 -dm 768 -s adnexus
python3 start.py -w 4 -ni 10 -bs 1000 -pa outputs/basic -d wikitext -e 8 -ec 2 -dm 768 -cp RAND -s adfabricus
python3 start.py -w 4 -ni 10 -bs 1000 -pa outputs/basic -d wikitext -e 8 -ec 2 -dm 768 -cp MTU -s adfabricus