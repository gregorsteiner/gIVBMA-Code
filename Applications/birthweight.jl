

using DataFrames, CSV, InvertedIndices, Random, LinearAlgebra, Distributions
using StatsModels, StatsPlots
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

# load data
df = CSV.read("birthweight.csv", DataFrame, drop = [1], missingstring="NA")

# drop all observations with missing values
dropmissing!(df)

# create model objects
y = df.bwght
x = df.cigs
formula = @formula(cigs ~ cigprice * cigtax * fatheduc * motheduc * parity * male * white)
Z = modelmatrix(formula, df)


# fit model
iters = 10000
res_pln = ivbma(y, x, Z; iter = iters, burn = Int(iters/5), dist = ["PLN", "PLN"], g_prior = "BRIC")
res_gauss = ivbma(log.(y), x, Z; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "PLN"], g_prior = "BRIC")

p = density([res_pln.τ res_gauss.τ], fill = true, alpha = 0.7,
            label = ["Poisson" "Gaussian"], xlabel = "τ", ylabel = "Density")
savefig(p, "Posterior_Birthweight.pdf")

# check instruments
ind_pln = sortperm(mean(res_pln.M, dims = 1)[1,:], rev = true)
ind_gauss = sortperm(mean(res_gauss.M, dims = 1)[1,:], rev = true)

formula.rhs[ind_pln][1:5]
formula.rhs[ind_gauss][1:5]

