

using DataFrames, CSV, InvertedIndices, Random, LinearAlgebra 
using StatsModels, Distributions, ProgressBars
using CairoMakie, LaTeXStrings, KernelDensity

# the following line needs to be run when using the gIVBMA package for the first time
# using Pkg; Pkg.add(url="https://github.com/gregorsteiner/gIVBMA.jl.git")
using gIVBMA

include("../Simulations/bma.jl")
include("../Simulations/competing_methods.jl")

##### Load and prepare data #####
df = CSV.read("birthweight.csv", DataFrame, drop = [1], missingstring="NA")

# drop all observations with missing values
dropmissing!(df)

# create model objects
y = df.bwght
x = df.cigs
Z = modelmatrix(@formula(cigs ~ cigprice + cigtax + fatheduc + motheduc), df)
W = modelmatrix(@formula(cigs ~ (parity + male + white) + (cigprice + cigtax + fatheduc + motheduc) & (parity + male + white)), df)


##### Run analysis #####
iters = 5000
res_pln = givbma(y, x, Z, W; iter = iters, burn = Int(iters/5), dist = ["PLN", "PLN"], g_prior = "hyper-g/n")
res_gauss = givbma(log.(y), x, Z, W; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "PLN"], g_prior = "hyper-g/n")
res_bma = bma(log.(y), x[:, 1:1], W; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")

iters_ivbma = 10*iters
res_ivbma = ivbma_kl(log.(y), x, Z, W, log.(y), x, Z, W; s = iters_ivbma, b = Int(iters_ivbma/5))


# save plot of posteriors
function compute_posterior_density(sample)
    # Separate zeros and non-zeros
    zeros = sample[sample .== 0.0]
    nonzeros = sample[sample .!= 0.0]
    
    # Calculate proportions
    n_total = length(sample)
    prop_zero = length(zeros) / n_total
    prop_nonzero = length(nonzeros) / n_total
    
    kde_obj = kde(nonzeros)

    # Scale the density to reflect the proportion of non-zero values
    scaled_density = kde_obj.density #* prop_nonzero

    return (
        kde_obj.x,
        scaled_density,
        prop_zero
    )
end


p = Figure()
ax1, ax2 = (Axis(p[1, 1], xlabel = L"\tau", ylabel = ""), Axis(p[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}", ylabel = ""))

lines!(ax1, rbw(res_pln)[1], label = "gIVBMA (Poisson)")
lines!(ax1, rbw(res_gauss)[1], label = "gIVBMA (Gaussian)", color = Makie.wong_colors()[2])
lines!(ax1, rbw_bma(res_bma)[1], label = "BMA (Gaussian)", color = Makie.wong_colors()[3])

d = compute_posterior_density(res_ivbma.τ_full[:, 1])
lines!(ax1, d[1], d[2], color = Makie.wong_colors()[4], label = "IVBMA")
lines!(ax1, [0.0, 0.0], [0.0, 200], color = Makie.wong_colors()[4])

density!(ax2, map(x -> x[1, 2]/x[2, 2], res_pln.Σ), label = "gIVBMA (Poisson)", color = :transparent, strokecolor = Makie.wong_colors()[1], strokewidth = 1.5)
density!(ax2, map(x -> x[1, 2]/x[2, 2], res_gauss.Σ), label = "gIVBMA (Gaussian)", color = :transparent, strokecolor = Makie.wong_colors()[2], strokewidth = 1.5)
density!(ax2, res_ivbma.Σ[1, 2, :] ./ res_ivbma.Σ[2, 2, :], color = :transparent, strokecolor = Makie.wong_colors()[4], strokewidth = 1.5)

Legend(p[2, 1:2], ax1, orientation = :horizontal)
save("Posterior_Birthweight.pdf", p)


println("Proportion of zeros for IVBMA: " * string(round(d[3], digits = 3)))


# check instruments
[[repeat([missing], 4); mean(res_pln.L, dims = 2)] mean(res_pln.M, dims = 2)]
[[repeat([missing], 4); mean(res_gauss.L, dims = 2)] mean(res_gauss.M, dims = 2)]
[[repeat([missing], 4); res_ivbma.L_bar[Not(1:2)]] res_ivbma.M_bar']


##### LPS Comparison #####

function kfold_cv(y, X, Z, W; k=5)
    n = length(y)
    fold_size = Int(floor(n / k))
    meths = ["BMA", "gIVBMA (PLN)", "gIVBMA (Gaussian)", "IVBMA (KL)", "TSLS"]
    lps_store = zeros(k, length(meths))

    # Generate indices for each fold
    indices = collect(1:n)
    fold_indices = [indices[i:min(i+fold_size-1, n)] for i in 1:fold_size:n]

    for fold in ProgressBar(1:k)
        # Define the test and training indices for the current fold
        test_idx = fold_indices[fold]
        train_idx = setdiff(indices, test_idx)

        # Split the data
        y_train, X_train, Z_train, W_train = y[train_idx], X[train_idx, :], Z[train_idx, :], W[train_idx, :]
        y_test, X_test, Z_test, W_test = y[test_idx], X[test_idx, :], Z[test_idx, :], W[test_idx, :]

        # Fit the model on the training set
        fit_bma = bma(log.(y_train), X_train, W_train; g_prior = "hyper-g/n")
        fit_pln = givbma(y_train, X_train, Z_train, W_train; dist = ["PLN", "PLN"], g_prior = "hyper-g/n")
        fit_gauss = givbma(log.(y_train), X_train, Z_train, W_train; dist = ["Gaussian", "PLN"], g_prior = "hyper-g/n")

        # Compute LPS for the current test observations
        corr = mean(log.(y_test)) # add correction for the log-linear models to get them to the same scale

        lps_store[fold, :] = [
            lps_bma(fit_bma, log.(y_test), X_test, W_test) + corr,
            lps(fit_pln, y_test, X_test, Z_test, W_test),
            lps(fit_gauss, log.(y_test), X_test, Z_test, W_test) + corr,
            ivbma_kl(log.(y_train), X_train, Z_train, W_train, log.(y_test), X_test, Z_test, W_test).lps + corr,
            tsls(log.(y_train), X_train, Z_train, W_train, log.(y_test), X_test, W_test).lps + corr
        ]
    end

    return round.(lps_store, digits = 3)
end

res = kfold_cv(y, x, Z, W; k = 5)

function create_latex_table(res, methods)
    # Calculate means and standard deviations
    means = round.(mean(res, dims=1)[:], digits = 3)
    
    # Find the index of the lowest mean value
    min_index = argmin(means)
    
    # Start building the LaTeX table with booktabs style
    table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{lc}\n"
    table *= "\\toprule\n"
    table *= "Method & Mean LPS \\\\\n"
    table *= "\\midrule\n"
    
    # Fill in the table rows
    for i in eachindex(methods)
        mean_std = string(means[i])
        if i == min_index
            # Highlight the minimum value
            mean_std = "\\textbf{" * mean_std * "}"
        end
        table *= methods[i] * " & " * mean_std * " \\\\\n"
    end
    
    # Close the table
    table *= "\\bottomrule\n\\end{tabular}\n\\caption{The mean LPS calculated over each fold of the birthweight data in a 5-fold cross-validation procedure.}\n\\label{tab:5_fold_LPS}\n\\end{table}"
    
    return table
end

meths = ["BMA", "gIVBMA (PLN)", "gIVBMA (Gaussian)", "IVBMA (KL)", "TSLS"]
latex_table = create_latex_table(res, meths)
println(latex_table)
