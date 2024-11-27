

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
Z = modelmatrix(@formula(cigs ~ cigtax + cigprice + fatheduc + motheduc + parity + male + white), df)


# fit model
res_bric = ivbma(y, x, Z; dist = ["PLN", "PLN"], g_prior = "BRIC")

density(res_bric.τ)

mean(res_bric.L, dims = 1)
mean(res_bric.M, dims = 1)

exp(-0.005)

