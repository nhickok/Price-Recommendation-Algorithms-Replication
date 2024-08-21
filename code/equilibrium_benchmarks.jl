using Optim

"""
    collusive(qualities::Vector{Float64}, mc_arr::Vector{Float64}, demand_states::Vector{Float64}, μ::Float64=1/4)

Finds total profit maximizing prices given the qualities of the firms, the marginal costs of the firms, the demand states, and (optionally) the scale of the logit error μ. Returns the vector of collusive prices and the total profit.
"""
function collusive(qualities::Vector{Float64}, mc_arr::Vector{Float64}, demand_states::Vector{Float64}, μ::Float64=1/4)
    f(p) = sum(sum(demand.(Ref(p), Ref(qualities), demand_states) ./length(demand_states)) .* (p .- mc_arr))

    collusive_prices = Optim.minimizer(optimize(p->-f(p), mc_arr .+ μ /demand_states[1], LBFGS(); autodiff=:forward))

    return collusive_prices, f(collusive_prices)
end

"""
    bertrandNash(qualities::Vector{Float64}, mc_arr::Vector{Float64}, demand_states::Vector{Float64}, μ::Float64=1/4)

Finds the Bertrand-Nash equilibrium prices given the qualities of the firms, the marginal costs of the firms, the demand states, and (optionally) the scale of the logit error μ. Returns the vector of Bertrand-Nash prices, the total profit of the firms, and the difference between the prices in the last two iterations to check convergence.
"""
function bertrandNash(qualities::Vector{Float64}, mc_arr::Vector{Float64}, demand_states::Vector{Float64}, μ::Float64=1/4)
    if length(qualities) > 2
        @warn "This function only works for two players"
    end

    f_1(p...) = sum(demand.(Ref([p...]), Ref(qualities), demand_states) ./ length(demand_states))[1] * (p[1] - mc_arr[1])
    f_2(p...) = sum(demand.(Ref([p...]), Ref(qualities), demand_states) ./ length(demand_states))[2] * (p[2] - mc_arr[2])
    BR_1(p2) = Optim.minimizer(optimize(p1 -> -f_1(p1[1], p2), [mc_arr[1] + μ /demand_states[1]], LBFGS(); autodiff=:forward))[1]
    BR_2(p1) = Optim.minimizer(optimize(p2 -> -f_2(p1, p2[1]), [mc_arr[2] + μ /demand_states[1]], LBFGS(); autodiff=:forward))[1]

    p = mc_arr[1] + μ /demand_states[1]
    diff = 1e7
    t = 1
    while diff > 1e-5 && t < 10000
        p_next = BR_1(BR_2(p))
        diff = abs(p_next - p)
        p = p_next
        t += 1
    end

    p2 = BR_2(p)
    return [p, p2], f_1(p, p2) + f_2(p, p2), diff
end
