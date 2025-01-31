

using Distributions, LinearAlgebra
using BSON, ProgressBars

using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA

include("competing_methods.jl")
include("bma.jl")
include("MA2SLS.jl")


##### Define auxiliary functions #####

# generate the instruments coefficients
# Note that with p = 20 choosing c_M = 3/8 leads to an R^2 of approximately 0.1.
function gen_instr_coeff(p, c_M)
    res = zeros(p)
    for i in 1:p
        if i <= p/2
            res[i] = c_M * (1 - i/(p/2 + 1))^4
        end
    end
    return res
end

function gen_data_KO2010(n = 100, c_M = 3/8, τ = 0.1, p = 20, k = 10, c = 1/2)
    V = rand(MvNormal(zeros(p+k), I), n)'
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α = 1; γ = 1
    δ_Z = gen_instr_coeff(p, c_M)
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1
    β = [ones(Int(k/2)); zeros(Int(k/2))]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    y = α .+ τ * x .+ W * β + u[:,1]

    return (y=y, x=x, Z=Z, W=W)
end

function givbma_res(y, x, Z, W, y_h, x_h, Z_h, W_h; g_prior = "BRIC", two_comp = false)
    res = givbma(y, x, Z, W; g_prior = g_prior, two_comp = two_comp)
    lps_int = lps(res, y_h, x_h, Z_h, W_h)
    return (
        τ = mean(rbw(res)[1]),
        CI = quantile(rbw(res)[1], [0.025, 0.975]),
        lps = lps_int
    )
end

function bma_res(y, X, Z, y_h, X_h, Z_h; g_prior = "hyper-g/n")
    res = bma(y, X, Z; g_prior = g_prior)
    lps_int = lps_bma(res, y_h, X_h, Z_h)
    return (
        τ = mean(rbw_bma(res)[1]),
        CI = quantile(rbw_bma(res)[1], [0.025, 0.975]),
        lps = lps_int
    )
end

function sim_func(m, n; c_M = 3/8, τ = 0.1, p = 20, k = 10, c = 1/2)
    meths = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "gIVBMA (2C)", "IVBMA (KL)", "OLS", "TSLS", "OTSLS", "JIVE", "RJIVE", "Post-Lasso", "MATSLS"]

    squared_error_store = Matrix(undef, m, length(meths))
    bias_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths))
    lps_store = Matrix(undef, m, length(meths))

    for i in ProgressBar(1:m)
        d = gen_data_KO2010(n, c_M, τ, p, k, c)
        d_h = gen_data_KO2010(Int(n/5), c_M, τ, p, k, c)

        res = [
            bma_res(d.y, d.x, d.W, d_h.y, d_h.x, d_h.W; g_prior = "hyper-g/n"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "BRIC"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "hyper-g/n"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "hyper-g/n", two_comp = true),
            ivbma_kl(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W),
            ols(d.y, d.x, d.W, d_h.y, d_h.x, d_h.W),
            tsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            tsls(d.y, d.x, d.Z[:, 1:10], d.W[:, 1:5], d_h.y, d_h.x, d_h.W[:, 1:5]),
            jive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            rjive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            post_lasso(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            matsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W)
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


##### Run the simulation #####

m = 100
c_M = [1/8, 3/8] # In this setup we get a first stage R^2 ≈ 0.1 with c_M = 3/8 and R^2 ≈ 0.01 with c_M = 1/8.

res50 = map(c -> sim_func(m, 50; c_M = c), c_M)
res500 = map(c -> sim_func(m, 500; c_M = c), c_M)

bson("SimResKO2010.bson", Dict(:n50 => res50, :n500 => res500))


##### Create Latex table #####

res = BSON.load("SimResKO2010.bson")

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
    methods = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "gIVBMA (2C)", "IVBMA (KL)", "OLS", "TSLS", "OTSLS", "JIVE", "RJIVE", "Post-Lasso", "MATSLS"]

    # Start the LaTeX table
    table_str = "\\begin{table}\n\\centering\n\\begin{tabular}{l*{8}{r}}\n\\toprule\n"
    table_str *= " & \\multicolumn{8}{c}{n = 50} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{R^2_f = 0.01} & \\multicolumn{4}{c}{R^2_f = 0.1} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Populate rows for each method for n = 50 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_50_001[i, 1], best_50_001.rmse) * " & "
        table_str *= highlight(table_50_001[i, 2], best_50_001.bias) * " & "
        table_str *= highlight(table_50_001[i, 3], best_50_001.coverage) * " & "
        table_str *= highlight(table_50_001[i, 4], best_50_001.lps) * " & "

        table_str *= highlight(table_50_01[i, 1], best_50_01.rmse) * " & "
        table_str *= highlight(table_50_01[i, 2], best_50_01.bias) * " & "
        table_str *= highlight(table_50_01[i, 3], best_50_01.coverage) * " & "
        table_str *= highlight(table_50_01[i, 4], best_50_01.lps) * " \\\\\n"
    end

    # Midrule for clarity before starting the n = 500 part
    table_str *= "\\midrule\n"
    table_str *= " & \\multicolumn{8}{c}{n = 500} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{R^2_f = 0.01} & \\multicolumn{4}{c}{R^2_f = 0.1} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Populate rows for each method for n = 500 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_500_001[i, 1], best_500_001.rmse) * " & "
        table_str *= highlight(table_500_001[i, 2], best_500_001.bias) * " & "
        table_str *= highlight(table_500_001[i, 3], best_500_001.coverage) * " & "
        table_str *= highlight(table_500_001[i, 4], best_500_001.lps) * " & "

        table_str *= highlight(table_500_01[i, 1], best_500_01.rmse) * " & "
        table_str *= highlight(table_500_01[i, 2], best_500_01.bias) * " & "
        table_str *= highlight(table_500_01[i, 3], best_500_01.coverage) * " & "
        table_str *= highlight(table_500_01[i, 4], best_500_01.lps) * " \\\\\n"
    end

    # Finish the table
    table_str *= "\\bottomrule\n\\end{tabular}\n"
    table_str *= "\\caption{Simulation results with many weak instruments based on 100 simulated datasets. The best values in each column are printed in bold.}\n"
    table_str *= "\\label{tab:KO2010_Sim}\n\\end{table}"

    return table_str
end

# Generate and print the LaTeX table with stacked multicolumns
latex_table = make_stacked_multicolumn_table(res)
println(latex_table)