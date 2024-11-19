#!/bin/bash
# Run the Python script
python ./domain_expansion/convert_data.py \
-n sinD \
-d /mnt/space/data/sinD/validation_converted \
--raw_data_path /home/boqi/CoDriving/data/SinD/validation \
--num_workers 5 \
--overwrite

python ./domain_expansion/convert_data.py \
-n sinD \
-d /mnt/space/data/sinD/training_converted \
--raw_data_path /home/boqi/CoDriving/data/SinD/training \
--num_workers 6 \
--overwrite
