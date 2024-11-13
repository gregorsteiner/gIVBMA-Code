

using Distributions, LinearAlgebra, ProgressBars

include("bma.jl")
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA


"""
    Define auxiliary functions
"""

# logistic function
logit(x) = exp(x) / (1+exp(x))

# data generating function
function gen_data(n, c, tau = [-1/2, 1/2], p = 10)
    Z = rand(MvNormal(zeros(p), I), n)'
    ι = ones(n)

    α, τ, β = (0, tau, zeros(p))
    Γ, Δ = ([0, 0], [(ones(p) ./ (1:p)) (ones(p) ./ (1:p))])

    u = rand(MvNormal([0, 0, 0], [1 c c; c 1 c; c c 1]), n)'

    Q = ι * Γ' + Z * Δ + u[:, 2:3] 
    μ, r = (logit.(Q[:, 2]), 1)
    B_α, B_β = (μ * r, r * (1 .- μ))
    X_2 = [rand(Beta(B_α[i], B_β[i])) for i in eachindex(μ)]
    X = [Q[:, 1] X_2]

    y = α*ι + X * τ + Z * β + u[:, 1]

    return (y = y, X = X, Z = Z)
end

# wrapper function for the separate analysis
function ivbma_sep(y, X, Z; dist = ["Gaussian", "BL"], g_prior = "BRIC")
    res_1 = ivbma(y, X[:, 1], [X[:, 2] Z]; dist = dist[1:1], g_prior = g_prior)
    res_2 = ivbma(y, X[:, 2], [X[:, 1] Z]; dist = dist[2:2], g_prior = g_prior)

    return (τ = [res_1.τ res_2.τ], x = missing)
end

# functions to compute the performance measures
function squared_error(τ, true_tau)
    τ_hat = mean(τ; dims = 1)[1,:]
    return (τ_hat - true_tau)' * (τ_hat - true_tau)
end

function coverage(τ, true_tau)
    covg = Vector{Bool}(undef, length(true_tau))
    for i in eachindex(true_tau)
        ci = quantile(τ[:, i], [0.025, 0.975])
        covg[i] = ci[1] < true_tau[i] < ci[2] 
    end
    return covg
end

# Wrapper function that runs the simulation
function sim_func(m, n, c; tau = [-1/2, 1/2], p = 10)
    meths = [bma, ivbma, ivbma_sep]
    g_priors = ["BRIC", "hyper-g/n"]

    squared_error_store = Matrix(undef, m, length(meths) * length(g_priors))
    times_covered = zeros(length(meths) * length(g_priors), 2)

    for i in ProgressBar(1:m)
        d = gen_data(n, c, tau, p)

        res = map(
            (f, g_p) -> f(d.y, d.X, d.Z; dist = ["Gaussian", "BL"], g_prior = g_p),
            repeat(meths, length(g_priors)),
            repeat(g_priors, length(meths))
        )

        squared_error_store[i,:] = map(x -> squared_error(x.τ, tau), res)
        covg = map(x -> coverage(x.τ, tau), res)
        times_covered += reduce(vcat, covg')
    end

    rmse = sqrt.(mean(squared_error_store, dims = 1))
    return (RMSE = rmse, Coverage = times_covered ./ m)
end

"""
    Run simulation
"""

m, n = (500, 50)
c = [0.25, 0.75]

results_low = sim_func(m, n, c[1])
results_high = sim_func(m, n, c[2])

"""
    Create table with results
"""



