


using DataFrames, CSV, InvertedIndices, Random, LinearAlgebra, Distributions
using StatsModels, StatsPlots
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

data_path = joinpath("smoke.csv")
df = CSV.read(data_path, DataFrame; drop = [1])


# drop columns not needed for the analysis
select!(df, Not(["lincome", "lcigpric"]))

formula = @formula(cigs ~ educ * age * white * income * restaurn + age^2 + educ^2 + age^3 + educ^3)

# create model objects
y = Vector(df.cigs)
x = Vector(df.cigpric)
Z = modelmatrix(formula, df)

# fit models
iters = 5000
res_bric = ivbma(y, x, Z; dist = ["PLN", "Gaussian"], g_prior = "BRIC", iter = iters)
res_hg = ivbma(y, x, Z; dist = ["PLN", "Gaussian"], g_prior = "hyper-g/n", iter = iters)

# plot posterior densities
density([res_bric.τ res_hg.τ], fill = true, alpha = 0.7, label = ["BRIC" "hyper-g/n"])
plot([res_bric.τ res_hg.τ], label = ["BRIC" "hyper-g/n"])


res_hg.L
res_hg.M
res_hg.G

[mean(res_bric.M, dims = 1)' mean(res_hg.M, dims = 1)']
