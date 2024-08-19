#!/bin/bash

wget https://image-net.org/data/imagenet10k_eccv2010.tar 
mkdir ../data/imagenet10k
tar -xvf imagenet10k_eccv2010.tar -C ../data/imagenet10k/
python select_extra_imagenet_1k.py --drp ../data
python extract_extra_imagenet_1k.py --drp ../data
