#!/bin/bash
julia --project=APR -t 25 "$(dirname "$0")/analysis.jl"
julia --project=APR "$(dirname "$0")/gen_tables.jl"
julia --project=APR "$(dirname "$0")/gen_figures.jl"
julia --project=APR "$(dirname "$0")/gen_robustness_figures.jl"