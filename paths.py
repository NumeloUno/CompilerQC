from pathlib import Path

cwd = (Path(__file__).parent.absolute())
parameters_path = (cwd / "benchmark/parameters")
benchmark_results_path = (cwd / "benchmark/results")
database_path = (cwd / "reversed_engineering/database")
MLP_energy_path = (cwd / "objective_function/")