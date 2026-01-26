

using Distributions, LinearAlgebra, Random
using BSON

# the following line needs to be run when using the gIVBMA package for the first time
# using Pkg; Pkg.add(url="https://github.com/gregorsteiner/gIVBMA.jl.git")
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
    V[:, 2:2:end] .*= 100 # adjust scaling by multiplying every other column by 100
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α, γ = (1, 1)
    δ_Z = gen_instr_coeff(p, c_M) ./ repeat([1.0, 100.0], Int(p/2))
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1 ./ repeat([1.0, 100.0], Int(k/2))
    β = [ones(Int(k/2)); zeros(Int(k/2))] ./ repeat([1.0, 100.0], Int(k/2))

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    y = α .+ τ * x .+ W * β + u[:,1]

    return (y=y, x=x, Z=Z, W=W)
end

function givbma_res(y, x, Z, W, y_h, x_h, Z_h, W_h; g_prior = "BRIC", cov_prior = "IW", ω_a = 1.0,  two_comp = false)
    res = givbma(y, x, Z, W; g_prior = g_prior, two_comp = two_comp, cov_prior = cov_prior, ω_a = ω_a, iter = 1200, burn = 200)
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
    meths = ["BMA (hyper-g/n)", "gIVBMA (IW)", "gIVBMA (ω_a = 0.1)", "gIVBMA (IW, 2C)", "BayesHS", "TSLS", "O-TSLS", "JIVE", "RJIVE", "MATSLS", "IVBMA", "Post-LASSO"]

    tau_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths))
    lps_store = Matrix(undef, m, length(meths))

    pl_no_instruments = 0 # count how many times post-lasso does not select any instruments

    idx_switch = findfirst(==("IVBMA"), meths) - 1

    Threads.@threads for i in 1:m
        d = gen_data_KO2010(n, c_M, τ, p, k, c)
        d_h = gen_data_KO2010(Int(n/5), c_M, τ, p, k, c)

        res = [
            bma_res(d.y, d.x, d.W, d_h.y, d_h.x, d_h.W; g_prior = "hyper-g/n"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "hyper-g/n", cov_prior = "IW"),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω_a = 0.1),
            givbma_res(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; g_prior = "hyper-g/n", two_comp = true),
            hsiv(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W),
            tsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            tsls(d.y, d.x, d.Z[:, 1:10], d.W[:, 1:5], d_h.y, d_h.x, d_h.W[:, 1:5]),
            jive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            rjive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W),
            matsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W)
        ]

        tau_store[i, 1:idx_switch] = map(x -> x.τ, res)
        lps_store[i, 1:idx_switch] = map(x -> x.lps, res)
        times_covered[1:idx_switch] += map(x -> (x.CI[1] < τ < x.CI[2]), res)
    end

    for i in 1:m
        d = gen_data_KO2010(n, c_M, τ, p, k, c)
        d_h = gen_data_KO2010(Int(n/5), c_M, τ, p, k, c)

        res = [
            ivbma_kl(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.Z, d_h.W; extract_instruments = false),
            post_lasso(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W)
        ]

        tau_store[i, (idx_switch+1):end] = map(x -> x.τ, res)
        lps_store[i, (idx_switch+1):end] = map(x -> x.lps, res)

        # The coverage calculation has to be handled differently if no instruments are selected (just add a zero then)
        if res[end].no_instruments
            pl_no_instruments += 1 # count the number of times post-lasso does not select any instruments
            times_covered[(idx_switch+1):end] += [map(x -> (x.CI[1] < τ < x.CI[2]), res[1:(end-1)]); 0]
        else
            times_covered[(idx_switch+1):end] += map(x -> (x.CI[1] < τ < x.CI[2]), res)
        end

    end

    # We could potentially get an error here if Post-Lasso never selects any instruments, but this should not happen for m large enough
    mae = [median(skipmissing(abs.(tau_store[:, i] .- τ))) for i in eachindex(meths)]
    bias = [abs(median(skipmissing(tau_store[:, i])) - τ) for i in eachindex(meths)]
    lps = [mean(lps_store[:, 1:(end-1)], dims = 1) missing]
    cov = times_covered ./ [repeat([m], length(meths)-1); m - pl_no_instruments]

    return (MAE = mae, Bias = bias, Coverage = cov, LPS = lps, No_Instruments_PL = pl_no_instruments)
end

##### Run the simulation #####
m = 100
c_M = [1/8, 3/8] # In this setup we get a first stage R^2 ≈ 0.1 with c_M = 3/8 and R^2 ≈ 0.01 with c_M = 1/8.

Random.seed!(42)
res50 = map(c -> sim_func(m, 50; c_M = c), c_M)
res500 = map(c -> sim_func(m, 500; c_M = c), c_M)

bson("SimResKO2010.bson", Dict(:n50 => res50, :n500 => res500))


##### Create Latex table #####

res = BSON.load("SimResKO2010.bson")

# Helper function to format individual results into a LaTeX tabular format
function format_result(res)
    tab = [res.MAE res.Bias res.Coverage res.LPS']
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
    methods = ["BMA (hyper-\$g/n\$)", "gIVBMA (IW)", "gIVBMA (\$\\omega_a = 0.1\$)", "gIVBMA (IW, 2C)", "BayesHS", "TSLS", "O-TSLS", "JIVE", "RJIVE", "MATSLS", "IVBMA", "Post-LASSO"]
    
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
    table_str *= "\\caption{\\textbf{Many Weak Instruments:} MAE, bias, coverage, and mean LPS on 100 simulated datasets. The best values in each column are printed in bold. Post-Lasso only returns estimates for \$\\tau\$, but not for the other coefficients, so we cannot compute the LPS. When no instrument is selected, no effect estimates are provided, therefore we do not consider those cases. The number of times no instruments were selected  in the first stage is (from top-left to bottom-right): " * join(PL_frequencies, ", ") * ".}\n"
    table_str *= "\\label{tab:KO2010_Sim}\n\\end{table}"

    return table_str
end

# Generate and print the LaTeX table with stacked multicolumns
latex_table = make_stacked_multicolumn_table(res)
println(latex_table)



##### Mixing of the indicators #####

# create traceplot for a single simulated dataset
using CairoMakie, LaTeXStrings

Random.seed!(42)
d50 = gen_data_KO2010(50)
d500 = gen_data_KO2010(500)

res50 = givbma(d50.y, d50.x, d50.Z, d50.W; g_prior = "hyper-g/n", iter = 6000, burn = 1000)
res50_chol = givbma(d50.y, d50.x, d50.Z, d50.W; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω_a = 0.1, iter = 6000, burn = 1000)
res50_2c = givbma(d50.y, d50.x, d50.Z, d50.W; g_prior = "hyper-g/n", two_comp = true, iter = 6000, burn = 1000)
res500 = givbma(d500.y, d500.x, d500.Z, d500.W; g_prior = "hyper-g/n", iter = 6000, burn = 1000)
res500_chol = givbma(d500.y, d500.x, d500.Z, d500.W; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω_a = 0.1, iter = 6000, burn = 1000)
res500_2c = givbma(d500.y, d500.x, d500.Z, d500.W; g_prior = "hyper-g/n", two_comp = true, iter = 6000, burn = 1000)

fig = Figure()
ax = Axis(fig[1, 1], ylabel = L"Treatment model size$$", xlabel = L"Iteration$$", title = L"n = 50")
lines!(ax, sum(res50.M, dims = 1)[1, :], label = L"IW$$", alpha = 0.7)
lines!(ax, sum(res50_2c.M, dims = 1)[1, :], label = L"IW, 2C$$", alpha = 0.7)
lines!(ax, sum(res50_chol.M, dims = 1)[1, :], label = L"Cholesky ($\omega_a = 0.1$)", alpha = 0.7)

ax500 = Axis(fig[1, 2], ylabel = L"Treatment model size$$", xlabel = L"Iteration$$", title = L"n = 500")
lines!(ax500, sum(res500.M, dims = 1)[1, :], label = "M", alpha = 0.7)
lines!(ax500, sum(res500_2c.M, dims = 1)[1, :], label = "IW, 2C", alpha = 0.7)
lines!(ax500, sum(res500_chol.M, dims = 1)[1, :], label = "Cholesky", alpha = 0.7)

fig[2, :] = Legend(fig, ax, orientation = :horizontal)
save("Many_Weak_Instruments_Mixing.pdf", fig)