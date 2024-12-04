

using Distributions, LinearAlgebra, ProgressBars, BSON

include("bma.jl")
include("competing_methods.jl")

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
    Γ, Δ = ([0, 0], [(ones(p)) (ones(p))])
    
    Σ = c .^ abs.((1:3) .- (1:3)')
    u = rand(MvNormal([0, 0, 0], Σ), n)'

    Q = ι * Γ' + Z * Δ + u[:, 2:3] 
    μ, r = (logit.(Q[:, 2]), 1)
    B_α, B_β = (μ * r, r * (1 .- μ))
    X_2 = [rand(Beta(B_α[i], B_β[i])) for i in eachindex(μ)]
    X_1 = Q[:, 1]
    X = [X_1 X_2]

    y = α*ι + X * τ + Z * β + u[:, 1]

    return (y = y, X = X, Z = Z)
end


# wrapper functions for each method
function bma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "BRIC")
    res = bma(y, X, Z; g_prior = g_prior)
    lps_int = lps_bma(res, y_h, X_h, Z_h)
    return (
        τ = mean(res.τ; dims = 1)[1,:],
        CI = (
            quantile(res.τ[:, 1], [0.025, 0.975]),
            quantile(res.τ[:, 2], [0.025, 0.975])
        ),
        lps = lps_int
    )
end

function ivbma_res(y, X, Z, y_h, X_h, Z_h; dist = ["Gaussian", "Gaussian"], g_prior = "BRIC")
    res = ivbma(y, X, Z; dist = dist, g_prior = g_prior)
    lps_int = lps(res, y_h, X_h, Z_h)
    return (
        τ = map(mean, rbw(res)),
        CI = (
            quantile(rbw(res)[1], [0.025, 0.975]),
            quantile(rbw(res)[2], [0.025, 0.975])
        ),
        lps = lps_int
    )
end

function tsls(y, x, Z, y_h, x_h; level = 0.05)
    n = length(y)

    U = [ones(n) x]
    V = [ones(n) Z]  
    P_V = V * inv(V'V) * V'
    
    β_hat = inv(U' * P_V * U) * U' * P_V * y
    τ_hat = β_hat[2:end]

    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))

    cov_τ = (σ2_hat * inv(U' * P_V * U))[2:end, 2:end]
    
    ci = (
        τ_hat[1] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov_τ[1, 1]),
        τ_hat[2] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov_τ[2, 2])
    )

    # compute lps on holdout dataset
    U_h = [ones(length(y_h)) x_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (τ = τ_hat, CI = ci, lps = lps)
end

# functions to compute the performance measures
function squared_error_and_bias(τ_hat, true_tau)
    return (
        se = (τ_hat - true_tau)' * (τ_hat - true_tau),
        bias = ones(length(τ_hat))' * (τ_hat - true_tau)
    )
end


function coverage(ci, true_tau)
    covg = Vector{Bool}(undef, length(true_tau))
    for i in eachindex(true_tau)
        covg[i] = ci[i][1] < true_tau[i] < ci[i][2] 
    end
    return covg
end

# Wrapper function that runs the simulation
function sim_func(m, n, c; tau = [-1/2, 1], p = 10)
    meths = ["BMA (BRIC)", "BMA (hyper-g/n)", "IVBMA (BRIC)", "IVBMA (hyper-g/n)", "IVBMA (KL)", "TSLS"]

    squared_error_store = Matrix(undef, m, length(meths))
    bias_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths), 2)
    lps_store = Matrix(undef, m, length(meths))

    for i in ProgressBar(1:m)
        d = gen_data(n, c, tau, p)
        d_h = gen_data(Int(n/5), c, tau, p)

        res = [
            bma_res(d.y, d.X, d.Z, d_h.y, d_h.X, d_h.Z; g_prior = "BRIC"),
            bma_res(d.y, d.X, d.Z, d_h.y, d_h.X, d_h.Z; g_prior = "hyper-g/n"),
            ivbma_res(d.y, d.X, d.Z, d_h.y, d_h.X, d_h.Z; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "BRIC"),
            ivbma_res(d.y, d.X, d.Z, d_h.y, d_h.X, d_h.Z; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "hyper-g/n"),
            ivbma_kl(d.y, d.X, d.Z, d_h.y, d_h.X, d_h.Z),
            tsls(d.y, d.X, d.Z, d_h.y, d_h.X),
        ]

        calc = map(x -> squared_error_and_bias(x.τ, tau), res)
        squared_error_store[i,:] = map(x -> x.se, calc)
        bias_store[i,:] = map(x -> x.bias, calc)
        covg = map(x -> coverage(x.CI, tau), res)
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

# save results to reuse later
bson("SimResMultEnd.bson", Dict(:low => low_endog, :high => high_endog))
res = BSON.load("SimResMultEnd.bson")

low_endog = round.(res[:low], digits = 2)
high_endog = round.(res[:high], digits = 2)

# Define row names
row_names = [
    "BMA (BRIC)",
    "BMA (hyper-g/n)",
    "gIVBMA (BRIC)",
    "gIVBMA (hyper-g/n)",
    "IVBMA (KL)",
    "TSLS"
]

# Helper function to bold the best value
highlight(value, best_value) = value == best_value ? "\\textbf{$(value)}" : string(value)

# Determine the best values within each scenario
best_low_rmse = minimum(low_endog[:, 1])
best_low_bias = low_endog[argmin(abs.(low_endog[:, 2]))[1], 2]
best_low_covg_x1 = low_endog[argmin(abs.(low_endog[:, 3] .- 0.95))[1], 3]
best_low_covg_x2 = low_endog[argmin(abs.(low_endog[:, 4] .- 0.95))[1], 4]
best_low_lps = minimum(low_endog[:, 5])

best_high_rmse = minimum(high_endog[:, 1])
best_high_bias = high_endog[argmin(abs.(high_endog[:, 2]))[1], 2]
best_high_covg_x1 = high_endog[argmin(abs.(high_endog[:, 3] .- 0.95))[1], 3]
best_high_covg_x2 = high_endog[argmin(abs.(high_endog[:, 4] .- 0.95))[1], 4]
best_high_lps = minimum(high_endog[:, 5])

# Start building the LaTeX table as a string
table = "\\begin{table}[h!]\n\\centering\n"
table *= "\\begin{tabular}{lccccc}\n"
table *= "\\toprule\n"
table *= "\\multicolumn{6}{c}{Low Endogeneity} \\\\\n"
table *= "\\midrule\n"
table *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov. X1} & \\textbf{Cov. X2} & \\textbf{LPS} \\\\\n"
table *= "\\midrule\n"

# Populate rows for Low Endogeneity with highlighted best values
for i in eachindex(row_names)
    row = row_names[i] * " & "
    row *= highlight(low_endog[i, 1], best_low_rmse) * " & "
    row *= highlight(low_endog[i, 2], best_low_bias) * " & "
    row *= highlight(low_endog[i, 3], best_low_covg_x1) * " & "
    row *= highlight(low_endog[i, 4], best_low_covg_x2) * " & "
    row *= highlight(low_endog[i, 5], best_low_lps) * " \\\\\n"
    global table *= row
end

# Add a midrule to separate Low and High Endogeneity sections
table *= "\\midrule\n"
table *= "\\multicolumn{6}{c}{High Endogeneity} \\\\\n"
table *= "\\midrule\n"
table *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov. X1} & \\textbf{Cov. X2} & \\textbf{LPS} \\\\\n"
table *= "\\midrule\n"

# Populate rows for High Endogeneity with highlighted best values
for i in eachindex(row_names)
    row = row_names[i] * " & "
    row *= highlight(high_endog[i, 1], best_high_rmse) * " & "
    row *= highlight(high_endog[i, 2], best_high_bias) * " & "
    row *= highlight(high_endog[i, 3], best_high_covg_x1) * " & "
    row *= highlight(high_endog[i, 4], best_high_covg_x2) * " & "
    row *= highlight(high_endog[i, 5], best_high_lps) * " \\\\\n"
    global table *= row
end

# Close the table
table *= "\\bottomrule\n"
table *= "\\end{tabular}\n"
table *= "\\caption{Simulation results with two endogenous variables (one Gaussian and one Beta) based on 100 simulated datasets.}\n"
table *= "\\label{tab:SimResMultEndo}\n"
table *= "\\end{table}"

# Print the LaTeX code
println(table)

