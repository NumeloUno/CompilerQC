#!/bin/bash 
NUM_PARALLEL_JOBS=15
part=1
name=McForDatabaseWithCore2
number=2
CSV_FILE=/net/fermion/csba1344/CompilerQC/benchmark/parameters/run_${number}/csvs/${name}_part_${part}.csv
tail -n +2 ${CSV_FILE} | parallel --progress --colsep ',' -j${NUM_PARALLEL_JOBS} python "/net/fermion/csba1344/CompilerQC/benchmark/benchmark_optimization.py --yaml_path={1} --batch_size={2} --min_N={3} --max_N={4} --min_C={5} --max_C={6} --max_size={7} --problem_folder={8} --run=${number}"  
            