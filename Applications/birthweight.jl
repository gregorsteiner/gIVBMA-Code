

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