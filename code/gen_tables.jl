include("$(@__DIR__)/qlearning.jl")

using .QLearn, Printf, JLD2, CSV, DataFrames, Statistics

#Global arguments
#------------------------
#region
suffix = isempty(ARGS) ? "" : first(ARGS)
#endregion

#Functions
#------------------------
#region
function summary_stats(df, opt)
    open("$(@__DIR__)/../tables/table1_$(opt.β).tex", "w") do f
        write(f, "\\begin{tabular}{lcc}\n")
        write(f, "\\toprule[1.5pt]\n")
        write(f, "& Informative & Uninformative\\\\ \n")
        write(f, "\\midrule\n")
        write(f, "Average Revenue & $(@sprintf("%.2f", mean(df.plat_reward[df.info .== 1.]))) & $(@sprintf("%.2f", mean(df.plat_reward[df.info .== 0.])))\\\\\n")
        write(f, "Average Consumer Surplus & $(@sprintf("%.2f", mean(df.CW[df.info .== 1.]))) & $(@sprintf("%.2f", mean(df.CW[df.info .== 0.])))\\\\\n")
        write(f, "Average Collusion Level & $(@sprintf("%.2f", mean(df.alpha[df.info .== 1.]))) & $(@sprintf("%.2f", mean(df.alpha[df.info .== 0.])))\\\\\n")
        write(f, "Average Price (Firm 1) & $(@sprintf("%.2f", mean(df.price_1[df.info .== 1.]))) & $(@sprintf("%.2f", mean(df.price_1[df.info .== 0.])))\\\\\n")
        write(f, "Average Price (Firm 2) & $(@sprintf("%.2f", mean(df.price_2[df.info .== 1.]))) & $(@sprintf("%.2f", mean(df.price_2[df.info .== 0.])))\\\\\n")
        write(f, "\\bottomrule[1.5pt]\n")
        write(f, "\\end{tabular}\n")
    end

    return nothing
end
#endregion

#Generate results table
#------------------------
#region
df = CSV.read("$(@__DIR__)/../output/simulation_results_$suffix.csv.gz", DataFrame; buffer_in_memory = true)
@load "$(@__DIR__)/../output/simulation_options_$suffix.jld2" opt opt_norecs
summary_stats(df, opt)
#endregion