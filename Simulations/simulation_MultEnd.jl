

using Distributions, LinearAlgebra, ProgressBars, BSON

include("bma.jl")
include("competing_methods.jl")

using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA


"""
    Define auxiliary functions
"""
# logistic function
logit(x) = exp(x) / (1+exp(x))

# data generating function
function gen_data(n, c, τ)
    ι = ones(n)

    Z_10 = rand(MvNormal(zeros(10), I), n)'
    Z_15 = Z_10[:, 1:5] * [0.3, 0.5, 0.7, 0.9, 1.1] * ones(5)' + rand(Normal(0, 1), n, 5)
    Z = [Z_10 Z_15]

    Σ = c .^ abs.((1:3) .- (1:3)')
    u = rand(MvNormal([0, 0, 0], Σ), n)'

    Q = ι * [4, -1]' + Z[:, 1] * [2, -2]' + Z[:, 5] * [-1, 1]' + Z[:, 7] * [1.5, 1]' + Z[:, 11] *[1, 1]' + Z[:, 13] * [1/2, -1/2]' + u[:, 2:3] 

    X_1 = Q[:, 1]
    μ, r = (logit.(Q[:, 2]), 1)
    B_α, B_β = (μ * r, r * (1 .- μ))
    X_2 = [rand(Beta(B_α[i], B_β[i])) for i in eachindex(μ)]
    X = [X_1 X_2]

    y = ι + X * τ + u[:, 1]

    return (y = y, X = X, Z = Z)
end


# wrapper functions for each method
function bma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "BRIC")
    res = bma(y, X, Z; g_prior = g_prior, iter = 1200, burn = 200)
    lps_int = lps_bma(res, y_h, X_h, Z_h)
    return (
        τ = map(mean, rbw_bma(res)),
        CI = (
            quantile(rbw_bma(res)[1], [0.025, 0.975]),
            quantile(rbw_bma(res)[2], [0.025, 0.975])
        ),
        lps = lps_int
    )
end

function givbma_res(y, X, Z, y_h, X_h, Z_h; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "BRIC", target = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0])
    res = givbma(y, X, Z; dist = dist, g_prior = g_prior, iter = 1200, burn = 200)
    lps_int = lps(res, y_h, X_h, Z_h)

    # get posterior probability of the true treatment model
    posterior_probability_M = mean(mapslices(row -> row == target, res.M, dims=2))

    return (
        τ = map(mean, rbw(res)),
        CI = (
            quantile(rbw(res)[1], [0.025, 0.975]),
            quantile(rbw(res)[2], [0.025, 0.975])
        ),
        lps = lps_int,
        posterior_probability_M = posterior_probability_M,
        M_bar = mean(res.M, dims = 1),
        M_size_bar = mean(sum(res.M, dims = 2))
    )
end


# functions to compute the coverage
function coverage(ci, true_tau)
    covg = Vector{Bool}(undef, length(true_tau))
    for i in eachindex(true_tau)
        covg[i] = ci[i][1] < true_tau[i] < ci[i][2] 
    end
    return covg
end

# Wrapper function that runs the simulation
function sim_func(m, n; c = 2/3, tau = [1/2, -1/2])
    meths = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "IVBMA (KL)"]

    tau_store = zeros(m, 2, length(meths))
    times_covered = zeros(length(meths), 2)
    lps_store = Matrix(undef, m, length(meths))

    posterior_probability_M_store = zeros(m, 4)
    mean_model_size_M_store = zeros(m, 4)
    pip_M_store = zeros(m, 4, 15)

    for i in ProgressBar(1:m)
        y, X, Z = gen_data(n, c, tau)
        y_h, X_h, Z_h = gen_data(Int(n/5), c, tau)

        res = [
            bma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "hyper-g/n"),
            givbma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "BRIC"),
            givbma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "hyper-g/n"),
            ivbma_kl(y, X, Z, y_h, X_h, Z_h)
        ]

        tau_store[i,:, :] = reduce(hcat, map(x -> x.τ, res))
        covg = map(x -> coverage(x.CI, tau), res)
        times_covered += reduce(vcat, covg')
        lps_store[i,:] = map(x -> x.lps, res)

        posterior_probability_M_store[i, :] = reduce(vcat, map(x -> x.posterior_probability_M, res[2:4]))
        mean_model_size_M_store[i, :] = reduce(vcat, map(x -> x.M_size_bar, res[2:4]))
        pip_M_store[i, :, :] = reduce(vcat, map(x -> Matrix(x.M_bar), res[2:4]))
    end

    mae = mapslices(slice -> median(mapslices(tau_hat -> sum(abs.(tau_hat - tau)), slice, dims = 2)), tau_store, dims = [1,2])[1, 1, :]
    bias = mapslices(tau_hat -> sum(abs.(median(tau_hat, dims = 1)[1, :] - tau)), tau_store, dims = [1, 2])[1, 1, :]
    lps = mean(lps_store, dims = 1)[1, :]

    return (
        MAE = mae, Bias = bias,
        Coverage = times_covered ./ m, LPS = lps,
        Posterior_Probability_true_M = posterior_probability_M_store,
        PIP_M = pip_M_store,
        Mean_Model_Size_M = mean_model_size_M_store
    )
end

"""
    Run simulation
"""
m = 100

results_low = sim_func(m, 50)
results_high = sim_func(m, 500)


bson("SimResMultEnd.bson", Dict(:n50 => results_low, :n500 => results_high))

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
table *= "\\caption{Simulation results with two endogenous variables (one Gaussian and one Beta) based on 100 simulated datasets. The best values in each column are printed in bold.}\n"
table *= "\\label{tab:SimResMultEndo}\n"
table *= "\\end{table}"

# Print the LaTeX code
println(table)

