#!bin/#!/usr/bin/env bash

# downloading databases
# train
python3 main.py downloader -y --Dataset /home/amit/Data/OID --multiclasses 1 --type_csv train --classes Window Door Building --image_IsGroupOf 0 --image_IsDepiction 0 --image_IsInside 0

# validation
python3 main.py downloader -y --Dataset /home/amit/Data/OID --multiclasses 1 --type_csv validation --classes Window Door Building  --image_IsGroupOf 0 --image_IsDepiction 0 --image_IsInside 0
