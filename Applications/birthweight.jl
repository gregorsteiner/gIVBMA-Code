

using DataFrames, CSV, InvertedIndices, Random, LinearAlgebra, Distributions
using StatsModels, StatsPlots, ProgressBars
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

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
res_pln = ivbma(y, x, Z, W; iter = iters, burn = Int(iters/5), dist = ["PLN", "PLN"], g_prior = "hyper-g/n", ν = 3)
res_gauss = ivbma(log.(y), x, Z, W; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "PLN"], g_prior = "hyper-g/n", ν = 3)

p = plot([rbw(res_pln); rbw(res_gauss)], alpha = 0.7, linewidth = 2.5,
         label = ["Poisson" "Gaussian"], xlabel = "τ", ylabel = "Density")
savefig(p, "Posterior_Birthweight.pdf")


plot([res_pln.τ res_gauss.τ])

# check instruments
mean(res_pln.M, dims = 1)
mean(res_gauss.M, dims = 1)

ind_pln = sortperm(mean(res_pln.M, dims = 1)[1,:], rev = true)
ind_gauss = sortperm(mean(res_gauss.M, dims = 1)[1,:], rev = true)

formula.rhs[ind_pln][1:5]
formula.rhs[ind_gauss][1:5]


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
    meths = ["gIVBMA (PLNB)", "gIVBMA (Gaussian)", "IVBMA (KL)", "TSLS"]
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
        lps_store[fold, :] = [
            lps(fit_pln, y_test, X_test, Z_test, W_test),
            lps(fit_gauss, log.(y_test), X_test, Z_test, W_test),
            ivbma_kl(log.(y_train), X_train, Z_train, W_train, log.(y_test), X_test, Z_test, W_test).lps,
            tsls(log.(y_train), X_train, Z_train, W_train, log.(y_test), X_test, W_test).lps
        ]
    end

    return round.(lps_store, digits = 3)
end

res = kfold_cv(y, x, Z, W; k = 5)
mean(res, dims = 1)
