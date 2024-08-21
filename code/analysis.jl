using Distributed, DataFrames, CSV, JLD2, Statistics
addprocs(5, exeflags="-t 1")

@everywhere begin
include("$(@__DIR__)/qlearning.jl")
end

@everywhere begin
using .QLearn, ProgressMeter
end

#Functions
#--------------------
#region
function run_sims(opt::Options)
    #Each worker is always assigned the same simulations for replicability
    return @showprogress pmap(_ -> SimulationResult(qlearn!(LearningModel(opt))...), 1:opt.num_simulations; batch_size = ceil(opt.num_simulations / nworkers()))
end

function create_results_dataframe(results, opt)
    prices = hcat([result.price_history for result in results]...)'
    demand_states = vcat([result.demand_state_history for result in results]...)
    demand_state_indices = [findfirst(opt.demand_state_grid .== demand_state) for demand_state in demand_states]
    plat_rewards = vcat([result.platform_reward_history for result in results]...)

    quality_vec = fill(opt.quality, opt.num_firms)
    prices_transpose = copy(prices')
    CW = similar(demand_states)
    Threads.@threads for i in eachindex(CW)
        CW[i] = welfare(prices_transpose[:, i], quality_vec, demand_states[i])
    end

    if opt.version == "full_info_recs"
        info = true
        demand_state_indices = [findfirst(opt.demand_state_grid .== demand_state) for demand_state in demand_states]
        nash_prices = first.(opt.nash_prices[demand_state_indices])
        collusive_prices = first.(opt.collusive_prices[demand_state_indices])
        α_arr = (prices[:,1] .- nash_prices) ./ (collusive_prices .- nash_prices)
    elseif opt.version == "no_recs"
        info = false
        α_arr = (prices[:,1] .- opt.nash_price[1]) ./ (opt.collusive_price[1] - opt.nash_price[1])
    end

    df = DataFrame(:price_1 => prices[:, 1], :price_2=> prices[:, 2], 
        :state => demand_states, :plat_reward => plat_rewards, 
        :CW => CW, :alpha => α_arr)
    df.info .= info
    df.beta .= opt.β

    return df
end

function summarize_results(df, mult, spec)
    summary_df = combine(groupby(df, :info), :plat_reward => mean, :CW => mean, :alpha => mean, :price_1 => mean, :price_2 => mean)
    summary_df.mult .= mult
    summary_df.spec .= spec

    return summary_df
end
#endregion

#Main specification
#--------------------
#region
opt = Options("full_info_recs")
results = run_sims(opt)
df = create_results_dataframe(results, opt)

opt_norecs = Options("no_recs")
results_norecs = run_sims(opt_norecs)
df_norecs = create_results_dataframe(results_norecs, opt_norecs)

suffix = isempty(ARGS) ? "" : first(ARGS)
CSV.write("$(@__DIR__)/../output/simulation_results_$(suffix).csv.gz", vcat(df, df_norecs); compress = true)
@save "$(@__DIR__)/../output/simulation_options_$(suffix).jld2" opt opt_norecs
#endregion

#Robustness
#--------------------
#region
mult = .9
summary_df = summarize_results(vcat(df, df_norecs), 1., "baseline")

#Marginal cost
opt = Options("full_info_recs", mc = mult*1)
results = run_sims(opt)
df = create_results_dataframe(results, opt)

opt_norecs = Options("no_recs", mc = mult*1)
results_norecs = run_sims(opt_norecs)
df_norecs = create_results_dataframe(results_norecs, opt_norecs)

summary_df = vcat(summary_df, summarize_results(vcat(df, df_norecs), mult, "Marginal cost"))

#Quality
opt = Options("full_info_recs", quality = mult*2)
results = run_sims(opt)
df = create_results_dataframe(results, opt)

opt_norecs = Options("no_recs", quality = mult*2)
results_norecs = run_sims(opt_norecs)
df_norecs = create_results_dataframe(results_norecs, opt_norecs)

summary_df = vcat(summary_df, summarize_results(vcat(df, df_norecs), mult, "Quality"))

#Logit error scale
opt = Options("full_info_recs", μ=mult*1/4)
results = run_sims(opt)
df = create_results_dataframe(results, opt)

opt_norecs = Options("no_recs", μ=mult*1/4)
results_norecs = run_sims(opt_norecs)
df_norecs = create_results_dataframe(results_norecs, opt_norecs)

summary_df = vcat(summary_df, summarize_results(vcat(df, df_norecs), mult, "Logit error"))

#Discount factor
opt = Options("full_info_recs", δ=mult*.95)
results = run_sims(opt)
df = create_results_dataframe(results, opt)

opt_norecs = Options("no_recs", δ=mult*.95)
results_norecs = run_sims(opt_norecs)
df_norecs = create_results_dataframe(results_norecs, opt_norecs)

summary_df = vcat(summary_df, summarize_results(vcat(df, df_norecs), mult, "Discount factor"))

#Learning rate
opt = Options("full_info_recs", α=mult*0.15)
results = run_sims(opt)
df = create_results_dataframe(results, opt)

opt_norecs = Options("no_recs", α=mult*0.15)
results_norecs = run_sims(opt_norecs)
df_norecs = create_results_dataframe(results_norecs, opt_norecs)

summary_df = vcat(summary_df, summarize_results(vcat(df, df_norecs), mult, "Learning rate"))

#Exploration rate
opt = Options("full_info_recs", β=mult*5e-6)
results = run_sims(opt)
df = create_results_dataframe(results, opt)

opt_norecs = Options("no_recs", β=mult*5e-6)
results_norecs = run_sims(opt_norecs)
df_norecs = create_results_dataframe(results_norecs, opt_norecs)

summary_df = vcat(summary_df, summarize_results(vcat(df, df_norecs), mult, "Exploration rate"))

#Save robustness results
CSV.write("$(@__DIR__)/../output/robustness_results_$(suffix).csv.gz", summary_df; compress = true)
#endregion