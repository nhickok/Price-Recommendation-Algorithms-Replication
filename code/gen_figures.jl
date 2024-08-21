include("$(@__DIR__)/qlearning.jl")

using .QLearn, CSV, DataFrames, DataFramesMeta, JLD2, Plots, Printf, Statistics

#Global arguments
#-------------------
#region
suffix = isempty(ARGS) ? "" : first(ARGS)
default(fontfamily = "helvetica")
ENV["GKSwstype"] = "100" #Plot in headless server mode
#endregion

#Functions
#-------------------
#region
function price_plot(data, opt; save = true)
    info = opt.version == "full_info_recs"

    plot_df = @chain data begin
        @rsubset(:info == info)
        @by(:state, :price = mean(:price_1))
        @orderby(:state)
    end
    plot_df.collusive_price = info ? first.(opt.collusive_prices) : fill(first(opt.collusive_price), opt.num_demand_states) 
    plot_df.nash_price = info ? first.(opt.nash_prices) : fill(first(opt.nash_price), opt.num_demand_states) 

    plot_opts = Dict(:linecolor => :black, :xlabel => "Demand State", :ylabel => "Average Price", :ylims => (minimum(opt.price_grid), maximum(opt.price_grid)), :xticks => opt.demand_state_grid, :legend => false)
    plt = Plots.plot(plot_df.state, plot_df.price, label="Average Price (Firm 1)"; plot_opts...)
    Plots.plot!(plot_df.state, plot_df.collusive_price, linestyle=:dash, label="Full Info. Collusive Price"; plot_opts...)
    Plots.plot!(plot_df.state, plot_df.nash_price, linestyle=:dashdot, label="Full Info. Bertrand Nash Prices"; plot_opts...)

    save ? Plots.pdf(plt, "$(@__DIR__)/../figs/state_avgprice_$(opt.version)_$(opt.β)") : display(plt)

    return nothing
end

function welfare_plot(data, opt; save = true)
    plot_df = @chain data begin
        @by([:state, :info], :CW = mean(:CW))
        unstack(:info, :CW; renamecols = x -> "info_$x")
        @orderby(:state)
    end
    
    plot_opts = Dict(:linecolor => :black, :xlabel => "Demand State", :ylabel => "Consumer Surplus (\$)", :xticks => opt.demand_state_grid)
    plt = Plots.plot(plot_df.state, plot_df.info_true, label="Full Info."; plot_opts...)
    Plots.plot!(plot_df.state, plot_df.info_false, linestyle=:dash, label="No Info."; plot_opts...)

    save ? Plots.pdf("$(@__DIR__)/../figs/state_avgwelfare_$(opt.β)") : display(plt)

end

function collusion_plot(data, opt; save = true)
    plot_df = @chain data begin
        @by(:info, :alpha = mean(:alpha))
        @orderby(-:info)
    end

    plot_opts = Dict(:color => :navy, :fillopacity => .9, :ylims => (0,1.1*maximum(plot_df.alpha)), :legend => :false, :ylabel => "Average Collusion Level")
    plt = Plots.bar(["Full Info.", "No Info."], plot_df.alpha; plot_opts...)

    save ? Plots.pdf("$(@__DIR__)/../figs/avgcollusion_$(opt.β)") : display(plt)
end

function welfare_decomposition(data, opt; save = true)

    α_fullinfo = mean(data.alpha[data.info .== true])
    α_noinfo = mean(data.alpha[data.info .== false])

    #Decomposition exercise
    #Prices with adjustment by demand stage and original collusion level
    prices_stage1 = α_fullinfo * (opt.collusive_prices) + (1-α_fullinfo)*(opt.nash_prices)
    CW_stage1 = [welfare(p, repeat([opt.quality], opt.num_firms), demand_state) for (p, demand_state) in zip(prices_stage1, opt.demand_state_grid)]
    
    #Prices with adjustment by demand state and no info collusion level
    prices_stage2 = α_noinfo * (opt.collusive_prices) + (1-α_noinfo)*(opt.nash_prices)
    CW_stage2 = [welfare(p, repeat([opt.quality], opt.num_firms), demand_state) for (p, demand_state) in zip(prices_stage2, opt.demand_state_grid)]
    
    #Prices w/o adjustment by demand state and no info collusion level
    prices_stage3 = α_noinfo * (opt.collusive_price) + (1-α_noinfo)*(opt.nash_price)
    CW_stage3 = [welfare(p, repeat([opt.quality], opt.num_firms), demand_state) for (p, demand_state) in zip(repeat([prices_stage3], opt.num_demand_states), opt.demand_state_grid)]
    
    loss_total = mean(CW_stage3) - mean(CW_stage1)
    loss_collusion = mean(CW_stage2) - mean(CW_stage1)
    loss_demand_adjustment = mean(CW_stage3) - mean(CW_stage2)
    
    pct_collusion = 100 * loss_collusion / loss_total
    pct_demand_adjustment = 100 * loss_demand_adjustment / loss_total
    switch_pt = round(pct_collusion / 10, digits = 1)
    
    plt = "
    \\begin{tikzpicture}
    \\draw (0,0) -- (10,0);
    \\draw (0,-.4) -- (0,.4);
    \\draw ($switch_pt,-.4) -- ($switch_pt,.4);
    \\draw (10,-.4) -- (10,.4);
    \\draw [line width=1.25pt, decorate, decoration = {calligraphic brace}] (0,.42) --  ($switch_pt,.42) node[pos=0.5,above=0pt,black]{\$$(@sprintf("%.0f", pct_collusion))\\%\$};
    \\draw [line width=1.25pt, decorate, decoration = {calligraphic brace}] ($switch_pt,.42) --  (10,.42) node[pos=0.5,above=0pt,black]{\$$(@sprintf("%.0f", pct_demand_adjustment))\\%\$};
    \\draw [stealth-] ($(switch_pt / 2),0) -- ($(switch_pt / 2),-1) node[below, align=center,text width=40mm] {Welfare loss from higher collusion level};
    \\draw [stealth-] ($((10 + switch_pt) / 2),0) -- ($((10 + switch_pt) / 2),-1) node[below, align=center,text width=40mm] {Welfare loss from prices adjusting to the demand state};
    \\end{tikzpicture}
    "

    if save
        open("$(@__DIR__)/../figs/decomposition_$(opt.β).tex", "w") do f
            write(f, plt)
        end
    else
        print(plt)
    end
end
#endregion

#Generate figures
#-------------------
#region
#Load simulation results and options
data = CSV.read("$(@__DIR__)/../output/simulation_results_$(suffix).csv.gz", DataFrame; buffer_in_memory = true)
@load "$(@__DIR__)/../output/simulation_options_$(suffix).jld2" opt opt_norecs

#Plot average price by demand state for full and no info recommenders
scalefontsizes(1.5)
price_plot(data, opt)
price_plot(data, opt_norecs)
scalefontsizes()

#Plot average consumer welfare by demand state for full and no info recommenders
welfare_plot(data, opt)

#Plot average collusion level for full and no info recommenders
collusion_plot(data, opt)

#Decompose welfare change into component due to pricing by demand state and component due to higher collusion level
welfare_decomposition(data, opt)
#endregion
