

using Distributions, LinearAlgebra, ProgressBars, BSON

include("bma.jl")
include("competing_methods.jl")

# the following line needs to be run when using the gIVBMA package for the first time
# using Pkg; Pkg.add(url="https://github.com/gregorsteiner/gIVBMA.jl.git")
using gIVBMA

##### Define auxiliary functions #####
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
    meths = ["BMA (hyper-g/n)", "gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "IVBMA (KL)", "OLS", "TSLS", "O-TSLS", "MATSLS"]

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
            ivbma_kl(y, X, Z, y_h, X_h, Z_h),
            ols(y, X, Z, y_h, X_h, Z_h),
            tsls(y, X, Z, y_h, X_h, Z_h),
            tsls(y, X, Z[:, [1, 5, 7, 11, 13]], y_h, X_h, Z_h[:, [1, 5, 7, 11, 13]]),
            matsls(y, X, Z, y_h, X_h, Z_h)
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

##### Run simulation #####

m = 100

results_low = sim_func(m, 50)
results_high = sim_func(m, 500)

bson("SimResMultEnd.bson", Dict(:n50 => results_low, :n500 => results_high))

##### Create Latex table with results #####
res = BSON.load("SimResMultEnd.bson")

function format_result(res)
    tab = [res.MAE res.Bias res.Coverage res.LPS]
    return round.(tab, digits = 2)
end

res50 = format_result(res[:n50])
res500 = format_result(res[:n500])

# Define row names
row_names = [
    "BMA (hyper-g/n)",
    "gIVBMA (BRIC)",
    "gIVBMA (hyper-g/n)",
    "IVBMA (KL)",
    "OLS",
    "TSLS",
    "O-TSLS",
    "MATSLS"
]

# Helper function to bold the best value
highlight(value, best_value) = value == best_value ? "\\textbf{$(value)}" : string(value)

# Determine the best values within each scenario
best_low_mae = minimum(res50[:, 1])
best_low_bias = res50[argmin(abs.(res50[:, 2]))[1], 2]
best_low_covg_x1 = res50[argmin(abs.(res50[:, 3] .- 0.95))[1], 3]
best_low_covg_x2 = res50[argmin(abs.(res50[:, 4] .- 0.95))[1], 4]
best_low_lps = minimum(res50[:, 5])

best_high_mae = minimum(res500[:, 1])
best_high_bias = res500[argmin(abs.(res500[:, 2]))[1], 2]
best_high_covg_x1 = res500[argmin(abs.(res500[:, 3] .- 0.95))[1], 3]
best_high_covg_x2 = res500[argmin(abs.(res500[:, 4] .- 0.95))[1], 4]
best_high_lps = minimum(res500[:, 5])

# Start building the LaTeX table as a string
table = "\\begin{table}[H]\n\\centering\n"
table *= "\\begin{tabular}{lccccc}\n"
table *= "\\toprule\n"
table *= "\\multicolumn{6}{c}{\$n=50\$} \\\\\n"
table *= "\\midrule\n"
table *= " & \\textbf{MAE} & \\textbf{Bias} & \\textbf{Cov. X1} & \\textbf{Cov. X2} & \\textbf{LPS} \\\\\n"
table *= "\\midrule\n"

# Populate rows for Low Endogeneity with highlighted best values
for i in eachindex(row_names)
    row = row_names[i] * " & "
    row *= highlight(res50[i, 1], best_low_mae) * " & "
    row *= highlight(res50[i, 2], best_low_bias) * " & "
    row *= highlight(res50[i, 3], best_low_covg_x1) * " & "
    row *= highlight(res50[i, 4], best_low_covg_x2) * " & "
    row *= highlight(res50[i, 5], best_low_lps) * " \\\\\n"
    global table *= row
end

# Add a midrule to separate Low and High Endogeneity sections
table *= "\\midrule\n"
table *= "\\multicolumn{6}{c}{\$n=500\$} \\\\\n"
table *= "\\midrule\n"
table *= " & \\textbf{MAE} & \\textbf{Bias} & \\textbf{Cov. X1} & \\textbf{Cov. X2} & \\textbf{LPS} \\\\\n"
table *= "\\midrule\n"

# Populate rows for High Endogeneity with highlighted best values
for i in eachindex(row_names)
    row = row_names[i] * " & "
    row *= highlight(res500[i, 1], best_high_mae) * " & "
    row *= highlight(res500[i, 2], best_high_bias) * " & "
    row *= highlight(res500[i, 3], best_high_covg_x1) * " & "
    row *= highlight(res500[i, 4], best_high_covg_x2) * " & "
    row *= highlight(res500[i, 5], best_high_lps) * " \\\\\n"
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



##### Create graphs with selection results #####

using CairoMakie, LaTeXStrings


# True posterior probability and mean model size
function create_boxplots(X_1, Y_1, X_2, Y_2)
    labels = [L"gIVBMA (BRIC$$)", L"gIVBMA (hyper-$g/n$)", L"IVBMA ($X_1$)", L"IVBMA ($X_2$)"]
    titles = [L"\text{Posterior probability true model}", L"\text{Mean model size}"]
    
    n_categories = size(X_1, 2)
    positions = 1:n_categories
    
    fig = Figure()
    
    # First row
    ax1 = Axis(fig[1, 1], title=titles[1], xticklabelsvisible=false, ylabel = L"n = 50")
    ax2 = Axis(fig[1, 2], title=titles[2], xticklabelsvisible=false)
    
    # Second row
    # Second row - with rotated and smaller labels
    ax3 = Axis(fig[2, 1], 
        xticks = (positions, labels), 
        ylabel = L"n = 500", 
        xticklabelrotation = pi/4,  # 45-degree rotation
        xticklabelsize = 12)  # Smaller font size
    ax4 = Axis(fig[2, 2], 
        xticks = (positions, labels), 
        xticklabelrotation = pi/4,  # 45-degree rotation
        xticklabelsize = 12)
    
    # First row plots
    boxplot!(ax1, repeat(positions, inner=size(X_1, 1)), vec(X_1))
    boxplot!(ax2, repeat(positions, inner=size(Y_1, 1)), vec(Y_1))
    
    # Second row plots
    boxplot!(ax3, repeat(positions, inner=size(X_2, 1)), vec(X_2))
    boxplot!(ax4, repeat(positions, inner=size(Y_2, 1)), vec(Y_2))
    
    # Adjust layout
    fig[1:2, 1:2] = [ax1 ax2; ax3 ax4]
    
    return fig
end

save("MultEndSimulation_Selection_Results.pdf", create_boxplots(
    res[:n50].Posterior_Probability_true_M,
    res[:n50].Mean_Model_Size_M,
    res[:n500].Posterior_Probability_true_M,
    res[:n500].Mean_Model_Size_M
))


# Posterior Inclusion probabilities
function create_comparison_table(n50_data, n500_data)
    # Method names
    methods = ["gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "IVBMA (X1)", "IVBMA (X2)"]
    
    # Start the LaTeX table
    latex_output = """
    \\begin{table}[h]
    \\centering
    \\begin{tabular}{l*{8}{c}}
    \\toprule
    & \\multicolumn{2}{c}{gIVBMA (BRIC)} & \\multicolumn{2}{c}{gIVBMA (hyper-g/n)} & \\multicolumn{2}{c}{IVBMA (X1)} & \\multicolumn{2}{c}{IVBMA (X2)} \\\\
    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}
    Variable"""
    
    # Add sample sizes
    for _ in 1:4
        latex_output *= " & n=50 & n=500"
    end
    latex_output *= " \\\\\n\\midrule\n"
    
    # Add data rows
    for var in 1:15
        latex_output *= var ∈ [1, 5, 7, 11, 13] ? string("\$\\boldsymbol{Z_{", var) *"}}\$" : string("\$Z_{", var) *"}\$"
        
        for method in 1:4
            # n50 value
            median_val = round(median(n50_data[:, method, var]), digits=3)
            latex_output *= " & \$" * string(median_val) * "\$"
            
            # n500 value
            median_val = round(median(n500_data[:, method, var]), digits=3)
            latex_output *= " & \$" * string(median_val) * "\$"
        end
        
        latex_output *= " \\\\\n"
    end
    
    # Close the table
    latex_output *= """\\bottomrule
    \\end{tabular}
    \\caption{\\textbf{Multiple endogenous variables with correlated instruments:} Median treatment posterior inclusion probabilities across 100 simulated datasets. The instruments included in the true model are printed in bold. Note that IVBMA uses separate treatment models for the two endogenous variables \$X_1\$ and \$X_2\$.}
    \\label{tab:SimMultEnd_PIPs}
    \\end{table}
    """
    
    return latex_output
end
create_comparison_table(res[:n50].PIP_M, res[:n500].PIP_M) |> println