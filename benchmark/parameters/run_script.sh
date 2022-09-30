#!/bin/bash 
NUM_PARALLEL_JOBS=10 
part=5
CSV_FILE=csvs/EnergyForLHZGraphs_part_${part}.csv
tail -n +2 ${CSV_FILE} | parallel --progress --colsep ',' -j${NUM_PARALLEL_JOBS} \
python "../benchmark_optimization.py --yaml_path={1} --batch_size=50 \
--min_N={3} --max_N={4} --min_C={5} --max_C={6} --max_size={7} \
--problem_folder={8}" 
