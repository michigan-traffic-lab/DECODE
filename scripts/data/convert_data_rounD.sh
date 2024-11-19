#!/bin/bash
# Run the Python script
python ./domain_expansion/convert_data.py \
-n rounD \
-d /home/boqi/CoDriving/data/rounD/validation_converted \
--raw_data_path /home/boqi/CoDriving/data/rounD/validation \
--num_workers 4 \
--overwrite

# python ./domain_expansion/convert_data.py \
# -n rounD \
# -d /home/boqi/CoDriving/data/rounD/training_converted \
# --raw_data_path /home/boqi/CoDriving/data/rounD/training \
# --num_workers 6 \
# --overwrite
