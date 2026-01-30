

using Distributions, LinearAlgebra, Random
using BSON, ProgressBars

include("bma.jl")
include("competing_methods.jl")
include("aux_functions.jl")

# the following line needs to be run when using the gIVBMA package for the first time
# using Pkg; Pkg.add(url="https://github.com/gregorsteiner/gIVBMA.jl.git")
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


function givbma_flex(y, x, Z, y_h, x_h, Z_h; g_prior = "BRIC", cov_prior = "IW", ω = 1.0)
    res = givbma(y, x, Z; g_prior = g_prior, cov_prior = cov_prior, ω_a = ω, iter = 1200, burn = 200)
    lps_int = lps(res, y_h, x_h, Z_h)
    N_Z = extract_instruments(res.L, res.M)
    return (
        τ = mean(rbw(res)[1]),
        CI = quantile(rbw(res)[1], [0.025, 0.975]),
        lps = lps_int,
        N_Z = N_Z
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
    #meths = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "gIVBMA (BRIC, ω_a = 1)", "gIVBMA (hyper-g/n, ω_a = 1)", "IVBMA (KL)", "OLS", "TSLS", "O-TSLS", "sisVIVE"]
    meths = [
        "BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "gIVBMA (BRIC, ω_a = 0.1)", "gIVBMA (hyper-g/n, ω_a = 0.1)", 
        "gIVBMA (BRIC, ω_a = 1)", "gIVBMA (hyper-g/n, ω_a = 1)", "gIVBMA (BRIC, ω_a = 10)", "gIVBMA (hyper-g/n, ω_a = 10)",
        "BayesHS", "TSLS", "O-TSLS", "IVBMA", "sisVIVE"
        ]


    tau_store = Matrix(undef, m, length(meths))
    times_covered = zeros(length(meths))
    lps_store = Matrix(undef, m, length(meths))
    instruments =  Array{Float64}(undef, 4, 8 + 1, m) # We only need this for the gIVBMA variants and for IVBMA

    idx_switch = findfirst(==("IVBMA"), meths) - 1

    Threads.@threads for i in 1:m
    #for i in 1:m
        d = gen_data_Kang2016(n, τ, p, s, c)
        d_h = gen_data_Kang2016(Int(n/5), τ, p, s, c)

        res = [
            bma_res(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n"),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "BRIC", cov_prior = "IW"),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n", cov_prior = "IW"),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "BRIC", cov_prior = "Cholesky", ω = 0.1),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω = 0.1),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "BRIC", cov_prior = "Cholesky", ω = 1.0),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω = 1.0),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "BRIC", cov_prior = "Cholesky", ω = 10.0),
            givbma_flex(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω = 10.0),
            hsiv(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z),
            tsls(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z),
            tsls(d.y, d.x, d.Z[:, (s+1):p], d.Z[:, 1:s], d_h.y, d_h.x, d_h.Z[:, 1:s]),
        ]

        tau_store[i, 1:idx_switch] = map(x -> x.τ, res)
        times_covered[1:idx_switch] += map(x -> (x.CI[1] < τ < x.CI[2]), res)
        lps_store[i, 1:idx_switch] = map(x -> x.lps, res)
        instruments[:, 1:(end-1), i] = reduce(hcat, map(x -> instrument_probabilities(x.N_Z, p, s), res[2:9]))
    end

    # separate loop for all methods that are not multithread compatible
    for i in 1:m
        d = gen_data_Kang2016(n, τ, p, s, c)
        d_h = gen_data_Kang2016(Int(n/5), τ, p, s, c)

        res = [ivbma_kl(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z), sisVIVE(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z)]
        tau_store[i, (idx_switch+1):end] = map(x -> x.τ, res)
        times_covered[(idx_switch+1):end] += map(x -> (x.CI[1] < τ < x.CI[2]), res)
        lps_store[i, (idx_switch+1):end] = map(x -> x.lps, res)
        instruments[:, end, i] = instrument_probabilities(res[1].N_Z, p, s)
    end

    mae = mapslices(x -> median(abs.(x .- τ)), tau_store, dims = 1)
    bias = mapslices(x -> abs(median(x) - τ), tau_store, dims = 1)
    lps = mean(lps_store, dims = 1)
    pp_instruments = mean(instruments; dims = 3)[:, :, 1]

    return (MAE = mae, Bias = bias, Coverage = times_covered ./ m, LPS = lps, PP_N_Z = pp_instruments)
end

"""
    Run simulation
"""
m = 100
ss = [3, 6]

Random.seed!(42)
res50 = map(s -> sim_func(m, 50; s = s), ss)
res500 = map(s -> sim_func(m, 500; s = s), ss)

bson("SimResKang2016.bson", Dict(:n50 => res50, :n500 => res500))

"""
    Create tables with results.
"""
# Helper function to format individual results into a LaTeX tabular format
function format_result(res; type = "Performance")
    if type == "Performance"
        tab = vcat(res.MAE, res.Bias, res.Coverage', res.LPS)'
        return round.(tab, digits = 2)
    elseif type == "Instruments"
        return round.(res.PP_N_Z', digits = 2)
    end
end

# Create latex table
function make_latex_table(res, colnames, methods; type = "Performance")
    table_50_001 = format_result(res[:n50][1]; type = type)
    table_50_01 = format_result(res[:n50][2]; type = type)
    table_500_001 = format_result(res[:n500][1]; type = type)
    table_500_01 = format_result(res[:n500][2]; type = type)

    # Start the LaTeX table
    table_str = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{l*{8}{r}}\n\\toprule\n"
    table_str *= " & \\multicolumn{8}{c}{\$n = 50\$} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{\$s = 3\$} & \\multicolumn{4}{c}{\$s = 6\$} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & $(colnames[1]) & $(colnames[2]) & $(colnames[3]) & $(colnames[4]) "
    table_str *= "& $(colnames[1]) & $(colnames[2]) & $(colnames[3]) & $(colnames[4])  \\\\\n\\midrule\n"

    # Populate rows for each method for n = 50 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        for j in 1:4
            table_str *= string(table_50_001[i, j]) * " & "
        end
        for j in 1:4
            table_str *= string(table_50_01[i, j]) * (j == 4 ? "" : " & ")
        end
        table_str *= " \\\\\n"
    end

    # Midrule for clarity before starting the n = 500 part
    table_str *= "\\midrule\n"
    table_str *= " & \\multicolumn{8}{c}{\$n = 500\$} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{\$s = 3\$} & \\multicolumn{4}{c}{\$s = 6\$} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & $(colnames[1]) & $(colnames[2]) & $(colnames[3]) & $(colnames[4]) "
    table_str *= "& $(colnames[1]) & $(colnames[2]) & $(colnames[3]) & $(colnames[4])  \\\\\n\\midrule\n"

    # Populate rows for each method for n = 50 scenarios
    for i in eachindex(methods)
        table_str *= methods[i] * " & "
        for j in 1:4
            table_str *= string(table_500_001[i, j]) * " & "
        end
        for j in 1:4
            table_str *= string(table_500_01[i, j]) * (j == 4 ? "" : " & ")
        end
        table_str *= " \\\\\n"
    end

    # Finish the table
    table_str *= "\\bottomrule\n\\end{tabular}\n"
    table_str *= "\\caption{\\textbf{Invalid Instruments:} }\n"
    table_str *= "\\label{tab:}\n\\end{table}"

    return table_str
end


# create table with instrument selection performance
cols = ["\$0\$", "\$(0, p-s) \$", "\$p-s\$", "\$(p-s, p]\$"]
meths = ["gIVBMA (BRIC)", "gIVBMA (h-\$g/n\$)", "gIVBMA (BRIC, \$\\omega_a = 0.1\$)", "gIVBMA (h-\$g/n\$, \$\\omega_a = 0.1\$)", 
        "gIVBMA (BRIC, \$\\omega_a = 1\$)", "gIVBMA (h-\$g/n\$, \$\\omega_a = 1\$)", "gIVBMA (BRIC, \$\\omega_a = 10\$)", "gIVBMA (h-\$g/n\$, \$\\omega_a = 10\$)",
        "IVBMA"]
make_latex_table(res, cols, meths; type = "Instruments") |> println

# compute prior probs for comparison
include("../Priors/model_prior.jl")
probs = [instrument_prior(i, 10) for i in 0:10]
for s in [3, 6]
    round.([probs[1], sum(probs[2:(10-s)]), probs[10-s+1], sum(probs[(10-s+2):end])], digits = 3) |> println
end


# Create table with main performance measures
cols = ["\\textbf{MAE}", "\\textbf{Bias}", "\\textbf{Cov.}", "\\textbf{LPS}"]
meths = ["BMA (h-\$g/n\$)", "gIVBMA (BRIC)", "gIVBMA (h-\$g/n\$)", "gIVBMA (BRIC, \$\\omega_a = 0.1\$)", "gIVBMA (h-\$g/n\$, \$\\omega_a = 0.1\$)", 
        "gIVBMA (BRIC, \$\\omega_a = 1\$)", "gIVBMA (h-\$g/n\$, \$\\omega_a = 1\$)", "gIVBMA (BRIC, \$\\omega_a = 10\$)", "gIVBMA (h-\$g/n\$, \$\\omega_a = 10\$)",
        "BayesHS", "TSLS", "O-TSLS", "IVBMA", "sisVIVE"]
make_latex_table(res, cols, meths; type = "Performance") |> println





##### Mixing of the indicators #####

# create traceplot for a single simulated dataset
gd(n) = gen_data_Kang2016(n, 0.1, 10, 3, 1/2)

Random.seed!(42)
d50 = gd(50)
d500 = gd(500)

res50 = givbma(d50.y, d50.x, d50.Z; g_prior = "hyper-g/n", iter = 6000, burn = 1000)
res50_chol = givbma(d50.y, d50.x, d50.Z; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω_a = 0.1, iter = 6000, burn = 1000)
res500 = givbma(d500.y, d500.x, d500.Z; g_prior = "hyper-g/n", iter = 6000, burn = 1000)
res500_chol = givbma(d500.y, d500.x, d500.Z; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω_a = 0.1, iter = 6000, burn = 1000)


lines(sum(res50.L, dims = 1)[1, :])
lines(sum(res50.M, dims = 1)[1, :])
lines(extract_instruments(res50.L, res50.M))



using CairoMakie, LaTeXStrings

fig = Figure()

# --- Setup Data for looping (optional but cleaner) ---
results = [
    (res50, res50_chol, L"n = 50"),
    (res500, res500_chol, L"n = 500")
]

line_refs = []
for (i, (r, r_chol, t)) in enumerate(results)
    
    # Row 1: Number of Outcome Variables (L)
    ax_L = Axis(fig[1, i], title = "", 
                subtitle = t, ylabel = (i == 1 ? L"L" : ""))
    
    # We only need to label these once for the legend to find them
    l1 = lines!(ax_L, vec(sum(r.L, dims = 1)), label = L"IW$$", alpha = 0.7)
    l2 = lines!(ax_L, vec(sum(r_chol.L, dims = 1)), label = L"Cholesky ($\omega_a = 0.1$)", alpha = 0.7)
    
    # Save references from the first column to use for the legend
    if i == 1
        append!(line_refs, [l1, l2])
    end
    # Row 2: Number of Treatment Variables (M)
    ax_M = Axis(fig[2, i], title = "", 
                ylabel = (i == 1 ? L"M" : ""))
    lines!(ax_M, vec(sum(r.M, dims = 1)), alpha = 0.7)
    lines!(ax_M, vec(sum(r_chol.M, dims = 1)), alpha = 0.7)

    # Row 3: Number of Implied Instruments (extract_instruments)
    ax_Z = Axis(fig[3, i], title = "", 
                xlabel = L"Iteration$$", ylabel = (i == 1 ? L"$N_Z$" : ""))
    lines!(ax_Z, extract_instruments(r.L, r.M), alpha = 0.7)
    lines!(ax_Z, extract_instruments(r_chol.L, r_chol.M), alpha = 0.7)
end

# Add a single legend at the bottom
Legend(fig[4, :], line_refs, [L"IW$$", L"Cholesky ($\omega_a = 0.1$)"], 
       orientation = :horizontal)

display(fig)
save("Invalid_Instruments_Mixing.pdf", fig)