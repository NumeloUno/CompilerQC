from pathlib import Path

cwd = (Path(__file__).parent.absolute())
parameters_path = (cwd / "parameter_estimation/parameters")
benchmark_results_path = (cwd / "parameter_estimation/results")
database_path = (cwd / "reversed_engineering/database")