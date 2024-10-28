
using CSV, DataFrames, PrettyTables, Statistics, StatsPlots
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

include("../Simulations/competing_methods.jl")

# GDP data
gdp = CSV.read("EminentDomainGDP.csv", DataFrame)

y = Vector(gdp.y)
x = Vector{Float64}(gdp.d)
W = Matrix(gdp[:, startswith.(names(gdp), "x")])
W = W[:, Not(2, 50)] # remove the intercept
Z = Matrix(gdp[:, startswith.(names(gdp), "z")])
Z = Z[:, Not(37)] # remove redundant level


iters = 20000 # number of iterations
m = [size(W, 2)/2, size(Z, 2)/5] # prior mean model size

res = ivbma(y, x, Z, W; iter = iters, burn = Int(iters/5), pln = true, m = m)
res_2c = ivbma(y, x, Z, W; iter = iters, burn = Int(iters/5), pln = true, two_comp = true, m = m)

plot(res)
plot(res_2c)


# Classical estimators
res_ols = ols(y, x, W)
res_tsls = tsls(y, x, Z, W, y, x, W)
res_post_lasso = post_lasso(y, x, Z, W, y, x, W; sim = false)

coefs = map(x -> round.(median(x.τ), digits = 4), [res, res_2c, res_ols, res_tsls, res_post_lasso])
CIs = [map(x -> round.(quantile(x.τ, [0.025, 0.975]), digits = 4), [res, res_2c]); map(x -> round.(x.CI, digits = 4), [res_ols, res_tsls, res_post_lasso])]


meths = ["IVBMA", "IVBMA-2C", "OLS", "TSLS", "Post-LASSO"]
tab = [coefs CIs]
pretty_table(
    tab;
    header = ["Est. (Post. Med.)", "Interval (95%)"],
    row_labels = meths,
    backend = Val(:latex)
)


p = density([res.τ res_2c.τ], fill = true, alpha = 0.6, label = ["IVBMA" "IVBMA-2C"])
savefig(p, "ED_Results.pdf")



