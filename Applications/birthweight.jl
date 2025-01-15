

using DataFrames, CSV, InvertedIndices, Random, LinearAlgebra 
using StatsModels, Distributions, ProgressBars
using CairoMakie
using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA

include("../Simulations/bma.jl")

"""
    Fit full IVBMA models.
"""
# load data
df = CSV.read("birthweight.csv", DataFrame, drop = [1], missingstring="NA")

# drop all observations with missing values
dropmissing!(df)

# create model objects
y = df.bwght
x = df.cigs
Z = modelmatrix(@formula(cigs ~ cigprice + cigtax + fatheduc + motheduc), df)
W = modelmatrix(@formula(cigs ~ (parity + male + white) + (cigprice + cigtax + fatheduc + motheduc) & (parity + male + white)), df)


# fit model
iters = 5000
res_pln = givbma(y, x, Z, W; iter = iters, burn = Int(iters/5), dist = ["PLN", "PLN"], g_prior = "hyper-g/n")
res_gauss = givbma(log.(y), x, Z, W; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "PLN"], g_prior = "hyper-g/n")
res_bma = bma(log.(y), x[:, 1:1], W; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")

# save plot of posteriors
p = Figure()
ax = Axis(p[1, 1], xlabel = "τ", ylabel = "Density")
CairoMakie.lines!(
    ax, rbw(res_pln)[1],
    color = Makie.wong_colors()[1],
    label = "gIVBMA (Poisson)"
    )
CairoMakie.lines!(
    ax, rbw(res_gauss)[1],
    color = Makie.wong_colors()[2],
    label = "gIVBMA (Gaussian)"
    )
CairoMakie.lines!(
    ax, rbw_bma(res_bma)[1],
    color = Makie.wong_colors()[3],
    label = "BMA (Gaussian)"
    )
axislegend()
save("Posterior_Birthweight.pdf", p)


# check instruments
mean(res_pln.M, dims = 1)
mean(res_gauss.M, dims = 1)


# check endogeneity
map(x -> x[1, 2]/x[2,2], res_pln.Σ) |> density
map(x -> x[1, 2]/x[2,2], res_gauss.Σ) |> density


"""
    LPS analysis.
"""

include("../Simulations/competing_methods.jl")

function kfold_cv(y, X, Z, W; k=5)
    n = length(y)
    fold_size = Int(floor(n / k))
    meths = ["gIVBMA (PLN)", "gIVBMA (Gaussian)", "IVBMA (KL)", "TSLS"]
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
        fit_pln = ivbma(y_train, X_train, Z_train, W_train; dist = ["PLN", "PLN"], g_prior = "hyper-g/n")
        fit_gauss = ivbma(log.(y_train), X_train, Z_train, W_train; dist = ["Gaussian", "PLN"], g_prior = "hyper-g/n")

        # Compute LPS for the current test observations
        corr = mean(log.(y_test)) # add correction for the log-linear models to get them to the same scale
        lps_store[fold, :] = [
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

methods = ["gIVBMA (PLN)", "gIVBMA (Gaussian)", "IVBMA (KL)", "TSLS"]
latex_table = create_latex_table(res, methods)
println(latex_table)
