To install the necessary dependencies, first activate and then instantiate the Julia environment. To do so, from the code directory simply run `julia --project=APR` and then instantiate with `]instantiate`. For replication purposes, Julia version 1.9.4 should be used as random number generation can differ across versions even with the same seed.

To generate the main output data as well as the tables and figures used in the paper, run the script `run_code.sh`. The script first runs `analysis.jl` which runs the main simulations and stores the necessary output. It then runs `gen_tables.jl`, `gen_figures.jl`, and `gen_robustness_figures.jl` to generate the tables and figures. It may be necessary to change the file permissions to make `run_code.sh` executable, e.g. by running `chmod +x run_code.sh`.

By default `analysis.jl` uses 25 threads and 5 worker processors. The number of threads can be changed using the `-t` flag in `run_code.sh`. The number of worker processors can be changed by adjusting `addprocs(...)` at line 2 in `analysis.jl`. For replication purposes, the default number of worker processors should be used.

