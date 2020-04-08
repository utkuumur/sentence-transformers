#!/usr/bin/bash

for (( i = 0; i < 2; i++ )); do
	python examples/cache_examples.py --data_dir /media/orion/Yeni Birim/patent_data/G/dataset/abstract/partitioned --output_dir /media/orion/Yeni Birim/patent_data/G/dataset/abstract/tokenized --model_name_or_path bert --n_threads 8 --part_no $i
done