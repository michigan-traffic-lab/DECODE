#!/bin/bash
# Run the Python script
python ./domain_expansion/convert_data.py \
-n inD \
-d /mnt/space/data/inD/validation_converted_new \
--raw_data_path /home/boqi/CoDriving/data/inD/validation \
--num_workers 4 \
--overwrite

python ./domain_expansion/convert_data.py \
-n inD \
-d /mnt/space/data/inD/training_converted_new \
--raw_data_path /home/boqi/CoDriving/data/inD/training \
--num_workers 6 \
--overwrite
