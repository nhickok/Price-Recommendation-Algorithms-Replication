#STRUCTS
#region
abstract type Agent end

mutable struct FirmState
    price_history::Vector{Vector{Int}}
    price_rec::Int
end

mutable struct PlatformState
    demand_state::Int
end

mutable struct Firm <: Agent
    mc::Float64
    quality::Float64
    p::Int
    Q::Array{Float64, 2}
    strat::Vector{Int}
    s::FirmState
end

mutable struct Platform <: Agent
    informative_rec::Bool
    s::PlatformState
end

mutable struct Environment
    price_grid::Vector{Float64}
    demand_states::Vector{Float64}

    δ::Float64
    μ::Float64
    demand_state_index::Int

    num_prices::Int
    num_demand_states::Int
    num_price_recs::Int
end

Environment(price_grid, demand_states, δ, μ, num_price_recs=length(price_grid)) = Environment(price_grid, demand_states, δ, μ, rand(rng, 1:length(demand_states)), length(price_grid), length(demand_states), num_price_recs)
#endregion

#FUNCTIONS
#region
"""
    deepcopy_state(state::FirmState)

Deep copy of the FirmState object.
"""
function deepcopy_state(state::FirmState)
    return FirmState(deepcopy(state.price_history), deepcopy(state.price_rec))
end

"""
    deepcopy_state(state::PlatformState)

Deep copy of the PlatformState object.
"""
function deepcopy_state(state::PlatformState)
    return PlatformState(deepcopy(state.demand_state))
end

"""
    updateDemandState!(e::Environment)

Draw a new demand state uniformly at random.
"""
function updateDemandState!(e::Environment)
    demand_state = rand(rng, 1:e.num_demand_states)
    e.demand_state_index = demand_state
end

"""
    observeDemandState!(p::Platform, demand_state::Int)

Platform observes the new demand state.
"""
function observeDemandState!(p::Platform, demand_state::Int)
    p.s.demand_state = demand_state
end

"""
    stateToIndex(p::Platform, e::Environment)

Converts the platform state to its unique index.
"""
function stateToIndex(p::Platform, e::Environment)
    return p.s.demand_state
end

"""
    getStrat(a::T, state_index::Int) where T <: Agent

Get the strategy of agent `a` at state `state_index`.
"""
function getStrat(a::T, state_index::Int) where T <: Agent
    return a.strat[state_index]
end

"""
    observePriceRec!(f::Firm, price_rec::Int)

Firm observes the price recommendation.
"""
function observePriceRec!(f::Firm, price_rec::Int)
    f.s.price_rec = price_rec
end

"""
    stateToIndex(f::Firm, e::Environment)

Converts the firm state to its unique index.
"""
function stateToIndex(f::Firm, e::Environment)
    index = 0
    multiplier = 1

    for prices in f.s.price_history
        for price in prices
            index += (price-1)*multiplier
            multiplier *= e.num_prices
        end
    end

    index += (f.s.price_rec-1)*multiplier
    multiplier *= e.num_price_recs

    return index + 1 #to make it one-indexed
end

"""
    updatePrice!(f::Firm, price::Int)

Update the price of firm `f`.
"""
function updatePrice!(f::Firm, price::Int)
    f.p = price
end

"""
    updatePriceHistory!(f::Firm, prices::Vector{Int})

Update the price history of firm `f`.
"""
function updatePriceHistory!(a::T, prices::Vector{Int}) where T <: Agent
    deleteat!(a.s.price_history, 1)
    push!(a.s.price_history, prices)
end

"""
    updateStrat!(a::T, row_index::Int) where T <: Agent

Update the strategy of agent `a` at state `row_index`.
"""
function updateStrat!(a::T, row_index::Int) where T <: Agent
    @views a.strat[row_index] = findmax(a.Q[row_index,:])[2]
end

"""
    demand(firms::Vector{Firm}, e::Environment)

Calculate the demand for each firm given the environment `e`.
"""
function demand(firms::Vector{Firm}, e::Environment)
    prices = [e.price_grid[f.p] for f in firms]
    qualities = [f.quality for f in firms]
    return demand(prices, qualities, e.demand_states[e.demand_state_index], e.μ)
end

"""
    demand(p_vec::Vector{T}, qual_vec::Vector{Float64}, demand_state::Float64, μ::Float64=1/4) where T <: Real

Calculate the demand for each firm given the prices, qualities, demand state, and (optionally) the scale of the logit error `μ`.
"""
function demand(p_vec::Vector{T}, qual_vec::Vector{Float64}, demand_state::Float64, μ::Float64=1/4) where T <: Real
    e_utilities = exp.((vcat([0.], qual_vec) .- demand_state .* vcat([0.],p_vec))./μ) #outside option has quality/price 0
    den = sum(e_utilities)
    return (e_utilities ./ den)[2:end]
end

"""
    welfare(firms::Vector{Firm}, e::Environment)

Calculate the total consumer welfare given firm strategies and the environment `e`.
"""
function welfare(firms::Vector{Firm}, e::Environment)
    prices = [e.price_grid[f.p] for f in firms]
    qualities = [f.quality for f in firms]
    return welfare(prices, qualities, e.demand_states[e.demand_state_index], e.μ)
end

"""
    welfare(p_vec::Vector{Float64}, qual_vec::Vector{Float64}, demand_state::Float64, μ::Float64=1/4)

Calculate the total consumer welfare given prices, qualities, demand state, and (optionally) the scale of the logit error `μ`.
"""
function welfare(p_vec::Vector{Float64}, qual_vec::Vector{Float64}, demand_state::Float64, μ::Float64=1/4)
    e_utilities = exp.((qual_vec .- demand_state .* p_vec) ./ μ)
    return μ * (1 + sum(e_utilities)) / demand_state #divide by price coeff. to put in monetary terms
end
#endregion