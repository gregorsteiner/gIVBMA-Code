

using Distributions, LinearAlgebra
using BSON, ProgressBars

include("bma.jl")
include("competing_methods.jl")

using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA


"""
    Define auxiliary functions
"""
function gen_data_Kang2016(n, τ, p, s, c)
    Z = rand(MvNormal(zeros(p), I), n)'

    α = γ = 1
    δ = ones(p) .* 5/32 # chosen s.t. the first-stage R^2 is approximately 0.2
    β = [ones(s); zeros(p-s)]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ Z * β + u[:,1]

    return (y=y, x=x, Z=Z)
end

function givbma_flex(y, x, Z, y_h, x_h, Z_h; g_prior)
    res = givbma(y, x, Z; g_prior = g_prior, iter = 1200, burn = 200)
    lps_int = lps(res, y_h, x_h, Z_h)
    return (
        τ = mean(rbw(res)[1]),
        CI = quantile(rbw(res)[1], [0.025, 0.975]),
        lps = lps_int
    )
end

function givbma_fix(y, x, Z, W, y_h, x_h, Z_h, W_h; g_prior = "hyper-g/n", two_comp = false)
    res = givbma(y, x, Z, W; g_prior = g_prior, two_comp = two_comp, iter = 1200, burn = 200)
    lps_int = lps(res, y_h, x_h, Z_h, W_h)
    return (
        τ = mean(rbw(res)[1]),
        CI = quantile(rbw(res)[1], [0.025, 0.975]),
        lps = lps_int
    )
end

function bma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "hyper-g/n")
    res = bma(y, X, Z; g_prior = g_prior, iter = 1200, burn = 200)
    lps_int = lps_bma(res, y_h, X_h, Z_h)
    return (
        τ = mean(rbw_bma(res)[1]),
        CI = quantile(rbw_bma(res)[1], [0.025, 0.975]),
        lps = lps_int
    )
end

function sim_func(m, n; τ = 0.1, p = 10, s = 2, c = 0.5)
    meths = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "O-gIVBMA (hyper-g/n)", "O-gIVBMA (2C)", "IVBMA (KL)", "OLS", "TSLS", "O-TSLS", "sisVIVE"]

    squared_error_store = Matrix(undef, m, length(meths))
    bias_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths))
    lps_store = Matrix(undef, m, length(meths))

    for i in ProgressBar(1:m)
        d = gen_data_Kang2016(n, τ, p, s, c)
        d_h = gen_data_Kang2016(Int(n/5), τ, p, s, c)

        res = [
            bma_res(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n"),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "BRIC"),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n"),
            givbma_fix(d.y, d.x, d.Z[:, (s+1):p], d.Z[:, 1:s], d_h.y, d_h.x, d_h.Z[:, (s+1):p], d_h.Z[:, 1:s]; g_prior = "hyper-g/n"),
            givbma_fix(d.y, d.x, d.Z[:, (s+1):p], d.Z[:, 1:s], d_h.y, d_h.x, d_h.Z[:, (s+1):p], d_h.Z[:, 1:s]; two_comp = true),
            ivbma_kl(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z),
            ols(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z),
            tsls(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z),
            tsls(d.y, d.x, d.Z[:, (s+1):p], d.Z[:, 1:s], d_h.y, d_h.x, d_h.Z[:, 1:s]),
            sisVIVE(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z)
        ]

        squared_error_store[i,:] = map(x -> (x.τ - τ)^2, res)
        bias_store[i,:] = map(x -> (x.τ - τ), res)
        times_covered += map(x -> (x.CI[1] < τ < x.CI[2]), res)
        lps_store[i, :] = map(x -> x.lps, res)
    end

    rmse = sqrt.(mean(squared_error_store, dims = 1))
    bias = mean(bias_store, dims = 1)
    lps = mean(lps_store, dims = 1)

    return (RMSE = rmse, Bias = bias, Coverage = times_covered ./ m, LPS = lps)
end

"""
    Run simulation
"""
m = 100
ss = [3, 6]

res50 = map(s -> sim_func(m, 50; s = s), ss)
res500 = map(s -> sim_func(m, 500; s = s), ss)

bson("SimResKang2016.bson", Dict(:n50 => res50, :n500 => res500))

"""
    Create Latex table
"""
res = BSON.load("SimResKang2016.bson")

# Helper function to format individual results into a LaTeX tabular format
function format_result(res)
    tab = vcat(res.RMSE, res.Bias, res.Coverage', res.LPS)'
    return round.(tab, digits = 2)
end

# Helper function to bold the best value within each scenario
highlight(value, best_value) = value == best_value ? "\\textbf{$(value)}" : string(value)

# Function to create the LaTeX table with table-specific best value highlighting and NA replacement
function make_stacked_multicolumn_table(res)
    # Extract tables for each scenario
    table_50_001 = format_result(res[:n50][1])
    table_50_01 = format_result(res[:n50][2])
    table_500_001 = format_result(res[:n500][1])
    table_500_01 = format_result(res[:n500][2])

    # Determine the best values within each table
    best_50_001 = (
        rmse = minimum(table_50_001[:, 1]),
        bias = table_50_001[argmin(abs.(table_50_001[:, 2])), 2],
        coverage = table_50_001[argmin(abs.(table_50_001[:, 3] .- 0.95)), 3],
        lps = minimum(table_50_001[:, 4])
    )
    best_50_01 = (
        rmse = minimum(table_50_01[:, 1]),
        bias = table_50_01[argmin(abs.(table_50_01[:, 2])), 2],
        coverage = table_50_01[argmin(abs.(table_50_01[:, 3] .- 0.95)), 3],
        lps = minimum(table_50_01[:, 4])
    )
    best_500_001 = (
        rmse = minimum(table_500_001[:, 1]),
        bias = table_500_001[argmin(abs.(table_500_001[:, 2])), 2],
        coverage = table_500_001[argmin(abs.(table_500_001[:, 3] .- 0.95)), 3],
        lps = minimum(table_500_001[:, 4])
    )
    best_500_01 = (
        rmse = minimum(table_500_01[:, 1]),
        bias = table_500_01[argmin(abs.(table_500_01[:, 2])), 2],
        coverage = table_500_01[argmin(abs.(table_500_01[:, 3] .- 0.95)), 3],
        lps = minimum(table_500_01[:, 4])
    )

    # Header for each method
    methods = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "O-gIVBMA (hyper-g/n)", "O-gIVBMA (2C)", "IVBMA (KL)", "OLS", "TSLS", "O-TSLS", "sisVIVE"]


    # Start the LaTeX table
    table_str = "\\begin{table}\n\\centering\n\\begin{tabular}{l*{8}{r}}\n\\toprule\n"
    table_str *= " & \\multicolumn{8}{c}{n = 50} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{s = 3} & \\multicolumn{4}{c}{s = 6} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Populate rows for each method for n = 50 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_50_001[i, 1], best_50_001.rmse) * " & "
        table_str *= highlight(table_50_001[i, 2], best_50_001.bias) * " & "
        
        # Replace 0 with NA in coverage column only for sisVIVE (last row, i == 5)
        cov_50_001 = (methods[i] == "sisVIVE" && table_50_001[i, 3] == 0) ? "NA" : highlight(table_50_001[i, 3], best_50_001.coverage)
        table_str *= cov_50_001 * " & "
        
        table_str *= highlight(table_50_001[i, 4], best_50_001.lps) * " & "
        table_str *= highlight(table_50_01[i, 1], best_50_01.rmse) * " & "
        table_str *= highlight(table_50_01[i, 2], best_50_01.bias) * " & "
        
        cov_50_01 = (methods[i] == "sisVIVE" && table_50_01[i, 3] == 0) ? "NA" : highlight(table_50_01[i, 3], best_50_01.coverage)
        table_str *= cov_50_01 * " & "
        
        table_str *= highlight(table_50_01[i, 4], best_50_01.lps) * " \\\\\n"
    end

    # Midrule for clarity before starting the n = 500 part
    table_str *= "\\midrule\n"
    table_str *= " & \\multicolumn{8}{c}{n = 500} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{s = 3} & \\multicolumn{4}{c}{s = 6} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Populate rows for each method for n = 500 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_500_001[i, 1], best_500_001.rmse) * " & "
        table_str *= highlight(table_500_001[i, 2], best_500_001.bias) * " & "
        
        cov_500_001 = (methods[i] == "sisVIVE" && table_500_001[i, 3] == 0) ? "NA" : highlight(table_500_001[i, 3], best_500_001.coverage)
        table_str *= cov_500_001 * " & "
        
        table_str *= highlight(table_500_001[i, 4], best_500_001.lps) * " & "
        table_str *= highlight(table_500_01[i, 1], best_500_01.rmse) * " & "
        table_str *= highlight(table_500_01[i, 2], best_500_01.bias) * " & "
        
        cov_500_01 = (methods[i] == "sisVIVE" && table_500_01[i, 3] == 0) ? "NA" : highlight(table_500_01[i, 3], best_500_01.coverage)
        table_str *= cov_500_01 * " & "
        
        table_str *= highlight(table_500_01[i, 4], best_500_01.lps) * " \\\\\n"
    end

    # Finish the table
    table_str *= "\\bottomrule\n\\end{tabular}\n"
    table_str *= "\\caption{Simulation results with s invalid instruments based on 100 simulated datasets. The best values in each column are printed in bold. The sisVIVE estimator does not provide any uncertainty quantification, so we set its coverage to NA.}\n"
    table_str *= "\\label{tab:Kang_Sim}\n\\end{table}"

    return table_str
end

# Generate and print the LaTeX table with stacked multicolumns
latex_table = make_stacked_multicolumn_table(res)
println(latex_table)

