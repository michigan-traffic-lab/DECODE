#!/bin/bash
# Run the Python script
python ./domain_expansion/convert_data.py \
-n inD \
-d ./data/inD/validation_converted \
--raw_data_path ./data/inD/validation \
--num_workers 4 \
--overwrite

python ./domain_expansion/convert_data.py \
-n inD \
-d ./data/inD/training_converted \
--raw_data_path ./data/inD/training \
--num_workers 6 \
--overwrite
