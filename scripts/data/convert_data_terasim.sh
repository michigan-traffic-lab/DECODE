#!/bin/bash
# Run the Python script
python ./domain_expansion/convert_data.py \
-n terasim \
-d /mnt/space/data/terasim/train_converted \
--raw_data_path /mnt/space/data/terasim/train \
--num_workers 1 \
--overwrite

