#STRUCTS
#region
struct LearningModel
    α::Float64
    β::Float64
    e::Environment
    firms::Vector{Firm}
    platform::Platform
end

struct SimulationResult
    LM::LearningModel
    t::Int
    price_history::Array{Float64, 2}
    price_rec_history::Vector{Float64}
    demand_state_history::Vector{Float64}
    platform_reward_history::Vector{Float64}
end

struct Options
    version::String

    num_simulations::Int
    α::Float64
    β::Float64

    mc::Float64
    quality::Float64
    δ::Float64
    memory::Int

    μ::Float64
    price_grid::Vector{Float64}
    demand_state_grid::Vector{Float64}

    num_firms::Int
    num_prices::Int
    num_price_recs::Int
    num_demand_states::Int
    num_firm_states::Int
    num_plat_states::Int

    Q_init_firm::Matrix{Float64}
    strat_init_firm::Vector{Int}
    state_init_firm::FirmState

    informative_recs::Bool
    state_init_plat::PlatformState

    collusive_prices::Vector{Vector{Float64}}
    nash_prices::Vector{Vector{Float64}}
    collusive_price::Vector{Float64}
    nash_price::Vector{Float64}
end
#endregion

#CONSTRUCTORS
#region
"""
    Options(version; num_simulations=1000, α=.15, β=5e-6, mc=1., quality=2., δ=.95, μ=1/4, demand_state_grid=collect(.8:.1:1.2), num_firms=2, memory=1)

Constructor for Options struct based on version label. Initializes collusive and Nash prices.
"""
function Options(version; num_simulations=1000, α=.15, β=5e-6, mc=1., quality=2., δ=.95, μ=1/4, demand_state_grid=collect(.8:.1:1.2), num_firms=2, memory=1)

    collusive_price, collusive_profit = collusive(repeat([quality], num_firms), repeat([mc], num_firms), demand_state_grid, μ)
    nash_price, nash_profit = bertrandNash(repeat([quality], num_firms), repeat([mc], num_firms), demand_state_grid, μ)

    collusive_prices = Vector{Vector{Float64}}(undef,length(demand_state_grid))
    nash_prices = Vector{Vector{Float64}}(undef,length(demand_state_grid))
    for (i,demand_state) in enumerate(demand_state_grid)
        prices, profit = collusive(repeat([quality], num_firms), repeat([mc], num_firms), [demand_state], μ)
        collusive_prices[i] = prices

        prices, profit, diff = bertrandNash(repeat([quality], 2), repeat([mc], 2), [demand_state], μ)
        nash_prices[i] = prices

    end
    num_firms == 2 ? min_p = floor(10*minimum(map(x->x[1], nash_prices)))/10 : min_p = floor(10*minimum(map(x->x[1], collusive_prices)))/10
    max_p = ceil(10*maximum(map(x->x[1], collusive_prices)))/10
    price_grid = collect(min_p:.1:max_p)

    num_demand_states = length(demand_state_grid)
    num_prices = length(price_grid)
    num_price_recs = num_demand_states
    num_firm_states = ((num_prices^num_firms)^memory)*(num_price_recs)
    num_plat_states = num_demand_states

    Q_init_firm = zeros(num_firm_states, num_prices)
    strat_init_firm = ones(num_firm_states)
    state_init_firm = FirmState(repeat([repeat([1], num_firms)], memory), 1)

    version == "full_info_recs" ? informative_recs = true : informative_recs = false
    state_init_plat = PlatformState(1)

    return Options(version, num_simulations, α, β, mc, quality, δ, memory, μ, price_grid, demand_state_grid, num_firms, num_prices, num_price_recs, num_demand_states, num_firm_states, num_plat_states, Q_init_firm, strat_init_firm, state_init_firm, informative_recs, state_init_plat, collusive_prices, nash_prices, collusive_price, nash_price)
end

"""
    LearningModel(opt::Options)

Constructor for LearningModel struct based on provided options.
"""
function LearningModel(opt::Options)
    e = Environment(opt.price_grid, opt.demand_state_grid, opt.δ, opt.μ)
    firms = [Firm(opt.mc, opt.quality, 1, copy(opt.Q_init_firm), copy(opt.strat_init_firm), deepcopy_state(opt.state_init_firm)) for i in 1:opt.num_firms]
    p = Platform(opt.informative_recs, deepcopy_state(opt.state_init_plat))

    return LearningModel(opt.α, opt.β, e, firms, p)
end
#endregion