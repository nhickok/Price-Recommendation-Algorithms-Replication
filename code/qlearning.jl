module QLearn

using Random, Distributed
import Future: randjump
const rng = randjump(Random.MersenneTwister(0), myid()*big(10)^20) #Sets random seed for each worker


export Environment, FirmState, PlatformState, Firm, Platform, LearningModel, Options, SimulationResult, qlearn!, welfare

include("$(@__DIR__)/pricing_rec_game.jl")
include("$(@__DIR__)/equilibrium_benchmarks.jl")
include("$(@__DIR__)/qlearning_structs.jl")

"""
    calculateRewards(firms::Vector{Firm}, e::Environment)

Calculates the rewards for the firms and platform given the current state of the environment.
"""
function calculateRewards(firms::Vector{Firm}, e::Environment)
    d = demand(firms, e)
    prices = [e.price_grid[f.p] for f in firms]
    mc = [f.mc for f in firms]

    return d'*prices, d .* (prices .- mc)
end

"""
    ϵ_greedy(a::Agent, e::Environment, ϵ::Float64)

Implements the ϵ-greedy action selection policy for an agent.
"""
function ϵ_greedy(a::Agent, e::Environment, ϵ::Float64)
    return ϵ_greedy(a.Q, stateToIndex(a, e), ϵ)
end

"""
    ϵ_greedy(Q::Array{Float64,2}, s::Int, ϵ::Float64)

Chooses a random action with probability ϵ, otherwise chooses the action with the highest Q-value.
"""
function ϵ_greedy(Q::Array{Float64,2}, s::Int, ϵ::Float64)
    if rand(rng) < ϵ
        return rand(rng, 1:size(Q)[2])
    else
        return findmax(Q[s,:])[2]
    end
end

"""
    updateQ!(a::Agent, e::Environment, α::Float64, state_ind::Int, p_ind::Int, r::Float64)

Updates the Q-value for a given state-action pair.
"""
function updateQ!(a::Agent, e::Environment, α::Float64, state_ind::Int, p_ind::Int, r::Float64)
    cont_val = maximum(a.Q[stateToIndex(a, e), :])
    a.Q[state_ind, p_ind] = (1-α)*a.Q[state_ind, p_ind] + α*(r + e.δ*cont_val)
end

"""
    qlearn!(LM::LearningModel; max_t=1000000000, max_stable_t=100000, noisy=false)

Runs the Q-learning algorithm for the given LearningModel.
"""
function qlearn!(LM::LearningModel; max_t=1000000000, max_stable_t=100000, noisy=false)
    t = 1
    stable_strat_counter = 0
    price_history = zeros(length(LM.firms), max_stable_t)
    price_rec_history = zeros(max_stable_t)
    demand_state_history = zeros(max_stable_t)
    platform_reward_history = zeros(max_stable_t)

    #Platform chooses initial price recommendation, observed by firms
    LM.platform.informative_rec ? price_rec = LM.e.demand_state_index : price_rec = rand(rng, 1:LM.e.num_demand_states)
    observePriceRec!.(LM.firms, price_rec)

    while stable_strat_counter < max_stable_t && t < max_t
        #Update threshold for choosing greedy action
        ϵ = exp(-LM.β*t)

        #Save demand state and firm states/relevant strategies before firm decisions
        demand_state = LM.e.demand_states[LM.e.demand_state_index]
        firm_states = stateToIndex.(LM.firms, Ref(LM.e))
        firm_strats = getStrat.(LM.firms, firm_states)

        #Firms choose prices
        firm_prices = ϵ_greedy.(LM.firms, Ref(LM.e), ϵ)
        updatePrice!.(LM.firms, firm_prices)

        #Gets reward from actions for platform and firms
        platform_reward, firm_rewards = calculateRewards(LM.firms, LM.e)

        #Update price histories
        updatePriceHistory!.(LM.firms, Ref(firm_prices))

        #Draw next period demand state, observed by platform
        updateDemandState!(LM.e)
        observeDemandState!(LM.platform, LM.e.demand_state_index)

        #Platform chooses price recommendation, observed by firms
        LM.platform.informative_rec ? price_rec = LM.e.demand_state_index : price_rec = rand(rng, 1:LM.e.num_demand_states)
        observePriceRec!.(LM.firms, price_rec)

        #Update Q-matrix now that next state is known
        updateQ!.(LM.firms, Ref(LM.e), LM.α, firm_states, firm_prices, firm_rewards)

        #Update firm strategies
        updateStrat!.(LM.firms, firm_states)

        #Check if strats stayed the same
        stable_strat = all(firm_strats .== getStrat.(LM.firms, firm_states))
        stable_strat ? stable_strat_counter += 1 : stable_strat_counter = 1

        price_history[:, stable_strat_counter] = [LM.e.price_grid[f.p] for f in LM.firms]
        price_rec_history[stable_strat_counter] = price_rec
        demand_state_history[stable_strat_counter] = demand_state
        platform_reward_history[stable_strat_counter] = platform_reward

        if noisy && t % 10000 == 0 println("Period $t, $stable_strat_counter stable periods") end

        t += 1
    end

    return LM, t, price_history, price_rec_history, demand_state_history, platform_reward_history
end
end
