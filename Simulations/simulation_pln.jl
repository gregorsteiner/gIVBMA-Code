
using Distributions, LinearAlgebra
using BSON, ProgressBars

using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA

include("competing_methods.jl")
include("bma.jl")
include("MA2SLS.jl")

##### Define auxiliary functions #####

function gen_instr_coeff(p::Integer, c_M::Number)
    res = zeros(p)
    for i in 1:p
        if i <= p/2
            res[i] = c_M * (1 - i/(p/2 + 1))^4
        end
    end
    return res
end


function gen_data(n::Integer = 100, c_M::Number = 3/8, τ::Number = 0.1, p::Integer = 20, k::Integer = 10, c::Number = 1/2)
    V = rand(MvNormal(zeros(p+k), I), n)'
    V[:, 2:2:end] .*= 100 # adjust scaling by multiplying every other column by 100
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α, γ = (1, 1)
    δ_Z = gen_instr_coeff(p, c_M) ./ repeat([1.0, 100.0], Int(p/2))
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1 ./ repeat([1.0, 100.0], Int(k/2))
    β = [ones(Int(k/2)); zeros(Int(k/2))] ./ repeat([1.0, 100.0], Int(k/2))

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    q = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    x = [rand(Poisson(exp(q[j]))) for j in eachindex(q)]
    y = α .+ τ * x .+ W * β + u[:,1]

    return (y=y, x=x, q=q, Z=Z, W=W)
end

function givbma_res(y, x, Z, W, y_h, x_h, Z_h, W_h; g_prior = "BRIC", two_comp = false)
    res = givbma(y, x, Z, W; g_prior = g_prior, two_comp = two_comp, dist = ["Gaussian", "PLN"], iter = 1200, burn = 200)
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


function sim_func(m, n; c_M = 3/8, τ = 0.1, p = 20, k = 10, c = 1/2)
    meths = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "gIVBMA (2C)", "IVBMA (KL)", "OLS", "TSLS", "O-TSLS", "JIVE", "RJIVE", "MATSLS", "Post-LASSO"]

    tau_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths))
    lps_store = Matrix(undef, m, length(meths))

    pl_no_instruments = 0 # count how many times post-lasso does not select any instruments

    for i in ProgressBar(1:m)
        d = gen_data(n, c_M, τ, p, k, c)
        d_h = gen_data(Int(n/5), c_M, τ, p, k, c)

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
            matsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            post_lasso(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W)
        ]

        tau_store[i,:] = map(x -> x.τ, res)
        lps_store[i, :] = map(x -> x.lps, res)

        # The coverage calculation has to be handled differently if no instruments are selected (just add a zero then)
        if res[end].no_instruments
            pl_no_instruments += 1 # count the number of times post-lasso does not select any instruments
            times_covered += [map(x -> (x.CI[1] < τ < x.CI[2]), res[1:(end-1)]); 0]
        else
            times_covered += map(x -> (x.CI[1] < τ < x.CI[2]), res)
        end
    end

    # We could potentially get an error here if Post-Lasso never selects any instruments, but this should not happen for m large enough
    mae = mapslices(x -> median(abs.(skipmissing(x) .- τ)), tau_store, dims=1)
    bias = mapslices(x -> median(skipmissing(x)) - τ, tau_store, dims=1)
    lps = [mean(lps_store[:, 1:(end-1)], dims = 1) missing]
    cov = times_covered ./ [repeat([m], length(meths)-1); m - pl_no_instruments]

    return (MAE = mae, Bias = bias, Coverage = cov, LPS = lps, No_Instruments_PL = pl_no_instruments)
end


##### Run the simulation #####

m = 100
c_M = [1/8, 3/8] # In this setup we get a first stage R^2 ≈ 0.1 with c_M = 3/8 and R^2 ≈ 0.01 with c_M = 1/8.

res50 = map(c -> sim_func(m, 50; c_M = c), c_M)
res500 = map(c -> sim_func(m, 500; c_M = c), c_M)

bson("SimResPLN.bson", Dict(:n50 => res50, :n500 => res500))


##### Create Latex table #####

res = BSON.load("SimResPLN.bson")

# Helper function to format individual results into a LaTeX tabular format
function format_result(res)
    tab = vcat(res.MAE, res.Bias, res.Coverage', res.LPS)'
    return round.(tab, digits = 2)
end

# Helper function to bold the best value within each scenario
highlight(value, best_value) = value == best_value ? "\\textbf{$(value)}" : string(value)

# Helper function to handle missing LPS values
function format_lps(lps_value)
    return ismissing(lps_value) ? "-" : string(round(lps_value, digits=2))
end

# Function to create the LaTeX table with table-specific best value highlighting and NA replacement
function make_stacked_multicolumn_table(res)
    # Extract tables for each scenario
    table_50_001 = format_result(res[:n50][1])
    table_50_01 = format_result(res[:n50][2])
    table_500_001 = format_result(res[:n500][1])
    table_500_01 = format_result(res[:n500][2])

    # Determine the best values within each table
    best_50_001 = (
        MAE = minimum(table_50_001[:, 1]),
        bias = table_50_001[argmin(abs.(table_50_001[:, 2])), 2],
        coverage = table_50_001[argmin(abs.(table_50_001[:, 3] .- 0.95)), 3],
        lps = minimum(filter(!ismissing, table_50_001[:, 4]))
    )
    best_50_01 = (
        MAE = minimum(table_50_01[:, 1]),
        bias = table_50_01[argmin(abs.(table_50_01[:, 2])), 2],
        coverage = table_50_01[argmin(abs.(table_50_01[:, 3] .- 0.95)), 3],
        lps = minimum(filter(!ismissing, table_50_01[:, 4]))
    )
    best_500_001 = (
        MAE = minimum(table_500_001[:, 1]),
        bias = table_500_001[argmin(abs.(table_500_001[:, 2])), 2],
        coverage = table_500_001[argmin(abs.(table_500_001[:, 3] .- 0.95)), 3],
        lps = minimum(filter(!ismissing, table_500_001[:, 4]))
    )
    best_500_01 = (
        MAE = minimum(table_500_01[:, 1]),
        bias = table_500_01[argmin(abs.(table_500_01[:, 2])), 2],
        coverage = table_500_01[argmin(abs.(table_500_01[:, 3] .- 0.95)), 3],
        lps = minimum(filter(!ismissing, table_500_01[:, 4]))
    )

    # Get the Post-Lasso frequencies of not selecting any instruments
    PL_frequencies = map(x -> x.No_Instruments_PL, [res[:n50][1], res[:n50][2], res[:n500][1], res[:n500][2]])

    # Header for each method
    methods = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "gIVBMA (2C)", "IVBMA (KL)", "OLS", "TSLS", "O-TSLS", "JIVE", "RJIVE", "MATSLS", "Post-LASSO"]
    
    # Start the LaTeX table
    table_str = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{l*{8}{r}}\n\\toprule\n"
    table_str *= " & \\multicolumn{8}{c}{\$n = 50\$} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{\$R^2_f = 0.01\$} & \\multicolumn{4}{c}{\$R^2_f = 0.1\$} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{MAE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{MAE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Populate rows for each method for n = 50 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_50_001[i, 1], best_50_001.MAE) * " & "
        table_str *= highlight(table_50_001[i, 2], best_50_001.bias) * " & "
        table_str *= highlight(table_50_001[i, 3], best_50_001.coverage) * " & "
        table_str *= highlight(format_lps(table_50_001[i, 4]), format_lps(best_50_001.lps)) * " & "

        table_str *= highlight(table_50_01[i, 1], best_50_01.MAE) * " & "
        table_str *= highlight(table_50_01[i, 2], best_50_01.bias) * " & "
        table_str *= highlight(table_50_01[i, 3], best_50_01.coverage) * " & "
        table_str *= highlight(format_lps(table_50_01[i, 4]), format_lps(best_50_01.lps)) * " \\\\\n"
    end

    # Midrule for clarity before starting the n = 500 part
    table_str *= "\\midrule\n"
    table_str *= " & \\multicolumn{8}{c}{\$n = 500\$} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{\$R^2_f = 0.01\$} & \\multicolumn{4}{c}{\$R^2_f = 0.1\$} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{MAE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{MAE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Populate rows for each method for n = 500 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        table_str *= highlight(table_500_001[i, 1], best_500_001.MAE) * " & "
        table_str *= highlight(table_500_001[i, 2], best_500_001.bias) * " & "
        table_str *= highlight(table_500_001[i, 3], best_500_001.coverage) * " & "
        table_str *= highlight(format_lps(table_500_001[i, 4]), format_lps(best_500_001.lps)) * " & "

        table_str *= highlight(table_500_01[i, 1], best_500_01.MAE) * " & "
        table_str *= highlight(table_500_01[i, 2], best_500_01.bias) * " & "
        table_str *= highlight(table_500_01[i, 3], best_500_01.coverage) * " & "
        table_str *= highlight(format_lps(table_500_01[i, 4]), format_lps(best_500_01.lps)) * " \\\\\n"
    end

    # Finish the table
    table_str *= "\\bottomrule\n\\end{tabular}\n"
    table_str *= "\\caption{\\textbf{Many Weak Instruments:} MAE, bias, coverage, and mean LPS with a Poisson endogenous variable and many weak instruments based on 100 simulated datasets. The best values in each column are printed in bold. Post-Lasso only returns estimates for \$\\tau\$, but not for the other coefficients, so we cannot compute the LPS. When no instrument is selected, no effect estimates are provided, therefore we do not consider those cases. The number of times no instruments were selected  in the first stage is (from top-left to bottom-right): " * join(PL_frequencies, ", ") * ".}\n"
    table_str *= "\\label{tab:PLN_Sim}\n\\end{table}"

    return table_str
end

# Generate and print the LaTeX table with stacked multicolumns
latex_table = make_stacked_multicolumn_table(res)
println(latex_table)