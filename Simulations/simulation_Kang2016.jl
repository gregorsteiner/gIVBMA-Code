

using Distributions, LinearAlgebra
using BSON, ProgressBars

include("competing_methods.jl")
include("sisVIVE.jl")

using Pkg; Pkg.activate("../../IVBMA")
using IVBMA


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

function ivbma_res(y, x, Z, y_h, x_h, Z_h; g_prior)
    res = ivbma(y, x, Z; g_prior = g_prior)
    lps_int = lps(res, y_h, x_h, Z_h)
    return (
        τ = mean(res.τ),
        CI = quantile(res.τ, [0.025, 0.975]),
        lps = lps_int
    )
end

function sim_func(m, n; τ = 0.1, p = 10, s = 2, c = 0.5)
    meths = ["IVBMA (BRIC)", "IVBMA (hyper-g/n)", "TSLS", "OTSLS", "sisVIVE"]

    squared_error_store = Matrix(undef, m, length(meths))
    bias_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths))
    lps_store = Matrix(undef, m, length(meths))

    for i in ProgressBar(1:m)
        d = gen_data_Kang2016(n, τ, p, s, c)
        d_h = gen_data_Kang2016(Int(n/5), τ, p, s, c)

        res = [
            ivbma_res(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "BRIC"),
            ivbma_res(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n"),
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
    methods = ["IVBMA (BRIC)", "IVBMA (hyper-g/n)", "TSLS", "O-TSLS", "sisVIVE"]

    # Start the LaTeX table
    table_str = "\\begin{table}\n\\centering\n\\begin{tabular}{l*{8}{r}}\n\\toprule\n"
    table_str *= " & \\multicolumn{8}{c}{n = 50} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{s = 3} & \\multicolumn{4}{c}{s = 6} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Populate rows for each method for n = 50 scenarios
    for i in 1:length(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_50_001[i, 1], best_50_001.rmse) * " & "
        table_str *= highlight(table_50_001[i, 2], best_50_001.bias) * " & "
        
        # Replace 0 with NA in coverage column only for sisVIVE (last row, i == 5)
        cov_50_001 = (i == 5 && table_50_001[i, 3] == 0) ? "NA" : highlight(table_50_001[i, 3], best_50_001.coverage)
        table_str *= cov_50_001 * " & "
        
        table_str *= highlight(table_50_001[i, 4], best_50_001.lps) * " & "
        table_str *= highlight(table_50_01[i, 1], best_50_01.rmse) * " & "
        table_str *= highlight(table_50_01[i, 2], best_50_01.bias) * " & "
        
        cov_50_01 = (i == 5 && table_50_01[i, 3] == 0) ? "NA" : highlight(table_50_01[i, 3], best_50_01.coverage)
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
    for i in 1:length(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_500_001[i, 1], best_500_001.rmse) * " & "
        table_str *= highlight(table_500_001[i, 2], best_500_001.bias) * " & "
        
        cov_500_001 = (i == 5 && table_500_001[i, 3] == 0) ? "NA" : highlight(table_500_001[i, 3], best_500_001.coverage)
        table_str *= cov_500_001 * " & "
        
        table_str *= highlight(table_500_001[i, 4], best_500_001.lps) * " & "
        table_str *= highlight(table_500_01[i, 1], best_500_01.rmse) * " & "
        table_str *= highlight(table_500_01[i, 2], best_500_01.bias) * " & "
        
        cov_500_01 = (i == 5 && table_500_01[i, 3] == 0) ? "NA" : highlight(table_500_01[i, 3], best_500_01.coverage)
        table_str *= cov_500_01 * " & "
        
        table_str *= highlight(table_500_01[i, 4], best_500_01.lps) * " \\\\\n"
    end

    # Finish the table
    table_str *= "\\bottomrule\n\\end{tabular}\n"
    table_str *= "\\caption{RMSE, bias, credible (or confidence) interval coverage (nominal 95\\%) and mean LPS (lower is better) on 100 simulated datasets. RMSE and Bias are based on the posterior mean of IVBMA.}\n"
    table_str *= "\\label{tab:Kang_Sim}\n\\end{table}"

    return table_str
end

# Generate and print the LaTeX table with stacked multicolumns
latex_table = make_stacked_multicolumn_table(res)
println(latex_table)

