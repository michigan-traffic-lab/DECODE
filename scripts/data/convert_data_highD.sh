#!/bin/bash
# Run the Python script
python ./domain_expansion/convert_data.py \
-n highD \
-d ./data/highD/validation_converted \
--raw_data_path ./data/highD/validation \
--num_workers 3 \
--num_files 3 \
--overwrite

python ./domain_expansion/convert_data.py \
-n highD \
-d ./data/highD/training_converted \
--raw_data_path ./data/highD/training \
--num_workers 10 \
--num_files 10 \
--overwrite
