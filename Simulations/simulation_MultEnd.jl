

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

    Σ = c .^ abs.((1:3) .- (1:3)')

    α, τ, β = (0, tau, zeros(p))
    Γ, Δ = ([0, 0], [(ones(p)) (ones(p))])

    u = rand(MvNormal([0, 0, 0], [1 c c; c 1 c; c c 1]), n)'

    Q = ι * Γ' + Z * Δ + u[:, 2:3] 
    μ, r = (logit.(Q[:, 2]), 1)
    B_α, B_β = (μ * r, r * (1 .- μ))
    X_2 = [rand(Beta(B_α[i], B_β[i])) for i in eachindex(μ)]
    X = [Q[:, 1] X_2]

    y = α*ι + X * τ + Z * β + u[:, 1]

    return (y = y, X = X, Z = Z)
end

# wrapper functions for each method
function bma_res(y, X, Z, y_h, X_h, Z_h; dist = ["Gaussian", "BL"], g_prior = "BRIC")
    res = bma(y, X, Z; g_prior = g_prior)
    lps_int = lps_bma(res, y_h, X_h, Z_h)
    return (τ = res.τ, lps = lps_int)
end

function ivbma_res(y, X, Z, y_h, X_h, Z_h; dist = ["Gaussian", "BL"], g_prior = "BRIC")
    res = ivbma(y, X, Z; dist = dist, g_prior = g_prior)
    lps_int = lps(res, y_h, X_h, Z_h)
    return (τ = res.τ, lps = lps_int)
end

function ivbma_sep(y, X, Z, y_h, X_h, Z_h; dist = ["Gaussian", "BL"], g_prior = "BRIC")
    res_1 = ivbma(y, X[:, 1], [X[:, 2] Z]; dist = dist[1:1], g_prior = g_prior)
    res_2 = ivbma(y, X[:, 2], [X[:, 1] Z]; dist = dist[2:2], g_prior = g_prior)

    lps_int = [
        lps(res_1, y_h, X_h[:, 1:1], [X_h[:, 2] Z_h])
        lps(res_2, y_h, X_h[:, 2:2], [X_h[:, 1] Z_h])
    ]

    return (τ = [res_1.τ res_2.τ], lps = mean(lps_int))
end

# functions to compute the performance measures
function squared_error_and_bias(τ, true_tau)
    τ_hat = mean(τ; dims = 1)[1,:]
    return (
        se = (τ_hat - true_tau)' * (τ_hat - true_tau),
        bias = ones(length(τ_hat))' * (τ_hat - true_tau)
    )
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
function sim_func(m, n, c; tau = [-1/2, 1], p = 10)
    meths = [bma_res, ivbma_res, ivbma_sep]
    g_priors = ["BRIC", "hyper-g/n"]

    squared_error_store = Matrix(undef, m, length(meths) * length(g_priors))
    bias_store = Matrix(undef, m, length(meths) * length(g_priors))
    times_covered = zeros(length(meths) * length(g_priors), 2)
    lps_store = Matrix(undef, m, length(meths) * length(g_priors))

    for i in ProgressBar(1:m)
        d = gen_data(n, c, tau, p)
        d_h = gen_data(Int(n/5), c, tau, p)

        res = map(
            (f, g_p) -> f(d.y, d.X, d.Z, d_h.y, d_h.X, d_h.Z; dist = ["Gaussian", "BL"], g_prior = g_p),
            repeat(meths, inner = length(g_priors)),
            repeat(g_priors, length(meths))
        )

        calc = map(x -> squared_error_and_bias(x.τ, tau), res)
        squared_error_store[i,:] = map(x -> x.se, calc)
        bias_store[i,:] = map(x -> x.bias, calc)
        covg = map(x -> coverage(x.τ, tau), res)
        times_covered += reduce(vcat, covg')
        lps_store[i,:] = map(x -> x.lps, res)
    end

    rmse = sqrt.(mean(squared_error_store, dims = 1))
    bias = mean(bias_store, dims = 1)
    lps = mean(lps_store, dims = 1)

    return (RMSE = rmse, Bias = bias, Coverage = times_covered ./ m, LPS = lps)
end

"""
    Run simulation
"""
m, n = (100, 100)
c = [0.3, 0.9]

results_low = sim_func(m, n, c[1])
results_high = sim_func(m, n, c[2])

"""
    Create table with results
"""

low_endog = [results_low.RMSE' results_low.Bias' results_low.Coverage results_low.LPS']
high_endog = [results_high.RMSE' results_high.Bias' results_high.Coverage results_high.LPS']

# Define row names
row_names = [
    "BMA (BRIC)",
    "BMA (hyper-g/n)",
    "IVBMA (BRIC)",
    "IVBMA (hyper-g/n)",
    "Sep. IVBMA (BRIC)",
    "Sep. IVBMA (hyper-g/n)"
]

# Start building the LaTeX table as a string
table = "\\begin{table}[h!]\n\\centering\n"
table *= "\\begin{tabular}{lccccc}\n"
table *= "\\toprule\n"
table *= "\\multicolumn{6}{c}{\\textbf{Low Endogeneity}} \\\\\n"
table *= "\\midrule\n"
table *= " & RMSE & Bias & Covg. X1 & Covg. X2 & LPS \\\\\n"
table *= "\\midrule\n"

# Populate rows for Low Endogeneity
for i in 1:6
    row = row_names[i] * " & "
    row *= join([string(round(low_endog[i, j], digits=2)) for j in 1:5], " & ") * " \\\\\n"
    table *= row
end

# Add a midrule to separate Low and High Endogeneity sections
table *= "\\midrule\n"
table *= "\\multicolumn{6}{c}{\\textbf{High Endogeneity}} \\\\\n"
table *= "\\midrule\n"
table *= " & RMSE & Bias & Covg. X1 & Covg. X2 & LPS \\\\\n"
table *= "\\midrule\n"

# Populate rows for High Endogeneity
for i in 1:6
    row = row_names[i] * " & "
    row *= join([string(round(high_endog[i, j], digits=2)) for j in 1:5], " & ") * " \\\\\n"
    table *= row
end

# Close the table
table *= "\\bottomrule\n"
table *= "\\end{tabular}\n"
table *= "\\caption{RMSE, bias, coverage, and mean log-predictive score (LPS) for low and high endogeneity scenarios over 100 simulated datasets.}\n"
table *= "\\label{tab:SimResMultEndo}\n"
table *= "\\end{table}"

# Print the LaTeX code
println(table)




