#!/bin/bash

cd ..
python3 start.py 4 1234 deepspeed bookcorpus standard
python3 start.py 4 1234 drop bookcorpus standard
python3 start.py 4 1234 demeter bookcorpus standard
python3 start.py 4 1234 adnexus bookcorpus standard
python3 start.py 4 1234 even_split bookcorpus standard
python3 start.py 4 1234 deepspeed wikitext standard
python3 start.py 4 1234 drop wikitext standard
python3 start.py 4 1234 demeter wikitext standard
python3 start.py 4 1234 adnexus wikitext standard
python3 start.py 4 1234 even_split wikitext standard
python3 start.py 4 1234 deepspeed sst2 standard
python3 start.py 4 1234 drop sst2 standard
python3 start.py 4 1234 demeter sst2 standard
python3 start.py 4 1234 adnexus sst2 standard
python3 start.py 4 1234 even_split sst2 standard