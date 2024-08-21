using CSV, DataFrames, DataFramesMeta, Plots, StatsPlots, Plots.PlotMeasures


#Global arguments
#----------------------------
#region
suffix = isempty(ARGS) ? "" : first(ARGS)
default(fontfamily = "helvetica")
ENV["GKSwstype"] = "100" #Plot in headless server mode
#endregion

#Functions
#----------------------------
#region
function robustness_plot(summary_df, outcome; save = true)
    elasticity_df = @chain summary_df begin
        @rsubset(:spec == "baseline")
        @orderby(-:info)
        select([:info, :mult, outcome])
        rename(:mult => :mult_baseline, outcome => :baseline)
        leftjoin(summary_df, _, on = :info)
        @rsubset(:spec != "baseline")
    end

    numerator = (elasticity_df[!, outcome] .- elasticity_df.baseline) ./ elasticity_df.baseline
    denom = (elasticity_df.mult .- elasticity_df.mult_baseline) ./ elasticity_df.mult_baseline
    elasticity_df.elasticity .= numerator ./ denom

    replace!(elasticity_df.spec, "Marginal cost" => "Marginal\ncost", 
        "Logit error" => "Logit \nscale", 
        "Discount factor" => "Discount\nfactor", 
        "Learning rate" => "Learning\nrate",
        "Exploration rate" => "Exploration\ndecay")

    plt = @df elasticity_df Plots.scatter(:spec, :elasticity, group = :info, label = ["Uninformative" "Informative"], xlabel = "Robustness Parameter", ylabel = "Elasticity", right_margin = 5mm, legend = :topright, color = [:black :maroon], dpi = 2400);

    #N.B. Png is necessary here because of a GR() bug that doesn't display negative signs properly in PDF output
    save ? Plots.png(plt, "$(@__DIR__)/../figs/robustness_$(outcome)") : display(plt)

    return nothing
end
#endregion

#Generate robustness figures
#----------------------------
#region
summary_df = CSV.read("$(@__DIR__)/../output/robustness_results_$suffix.csv.gz", DataFrame; buffer_in_memory = true)

robustness_plot(summary_df, :plat_reward_mean)
robustness_plot(summary_df, :CW_mean)
robustness_plot(summary_df, :alpha_mean)
robustness_plot(summary_df, :price_1_mean)
#endregion