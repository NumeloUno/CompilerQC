#!/bin/bash
find -name "*yaml" | parallel python ../../benchmark_optimization.py --yaml_path={} -b=10 -minN=10 -maxN=15 --min_C=10 --max_C=40 --max_size=1 --problem_folder='problems_by_square_density/fsquare_density_of_0.9'
