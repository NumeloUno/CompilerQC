To run benchmarks, you first have to create some settings to be benchmarked. Each setting is saved in a .yaml-file.
The settings can be easily created by running the create settings script. Here, you have to specify the settings you want to benchmark and set a number to identify the set of settings you created.
The script creates a folder run(number) in parameters. run/number) containts the settings and a csvs folder with csv files containing information about the benchmark (problem sizes, problem folders ...)
Furthermore in run(number) the folder sh scripts is created, these scripts has to be runned in order to start the benchmark.
The results are saved in the results folder. To evaluted the results, move to plot scripts folder and run create csvs of results.py (you have to change the number of the run accordingly). Open the visualize benchmarks notebook and visualize the sults.
