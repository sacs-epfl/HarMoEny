# HarMoEny
This reposity holds the code to HarMoEny _and_ the accompanying artifacts for Middleware'25. The code for HarMoEny can be found under `scr/harmonymoe`.

## Pre-Setup
If you are using an amazon EC2 instance, we recommend **p3dn.24xlarge**: which is the closest to our DGX1 machine. There is a setup script for your ec2 machine.
```bash
cd setup
./setup-ec2.sh
```  
This script will install all necessary packages and libraries **and** reboot the instance. If you are using your own server you may already have many of these already installed. 

## Setup
To run HarMoEny's artifacts, a Dockerfile, is provided, with all the necessary versions for each library. Please build the image with the following command.
```bash
./build_image.sh
```

## Running
Once the image is built you can start the image with `./start_image.sh`. After navigation to `cd experiments` you can find scripts to execute the various experiments and serve as inspiration for creating your own.

## Important
- If you want to run ExFlow you need to get a gurobi license; see `licenses/README.md`.
- HF will save to `/cache` inside the docker container. The default location is `../cache`, you can change it in `start_image.sh`.
