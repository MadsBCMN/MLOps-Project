import pstats
# run this line ine the script
# python -m cProfile -o output_file -s cumulative train_model.py

# Load the profiling data
stats = pstats.Stats("test/output_file")
stats.sort_stats("cumulative")

# Print the statistics
stats.print_stats()

# use snakeviz in comand line
# snakeviz test/output_file

#tottime (Total Time): Represents the total time spent in a function excluding time spent in subfunctions.

#cumtime (Cumulative Time): Represents the total time spent in a function including time spent in subfunctions.