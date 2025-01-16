
using DataFrames, CSV, Random, Statistics
using CairoMakie, LaTeXStrings, PrettyTables
using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA
include("../Simulations/bma.jl")

# Prepare data
d = CSV.read("wmales90.csv", DataFrame)

y = Vector(d.wage)
x = Vector(d.Educ)
col_names = ["Urban", "Exp", "Exp^2", "NC", "W", "S", "Ability", "Father_E", "Mother_E", "Married", "Unemploy"]
Z = Matrix(d[:, col_names])

# Run analysis
iters = 5000
res_bric = givbma(y, x, Z; iter = iters, burn = Int(iters/5), g_prior = "BRIC")
res_hg = givbma(y, x, Z; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")
res_bma = bma(y, x[:, 1:1], Z; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")

# Plot with posterior results
cols = Makie.wong_colors()

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"\tau")
lines!(ax1, rbw(res_bric)[1], color = cols[1], label = "gIVBMA (BRIC)")
lines!(ax1, rbw(res_hg)[1], color = cols[2], label = "gIVBMA (hyper-g/n)")
lines!(ax1, rbw_bma(res_bma)[1], color = cols[3], label = "BMA (hyper-g/n)")

ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax2, map(x -> x[1, 2]/x[2, 2], res_bric.Σ), color = :transparent, strokecolor = cols[1], strokewidth = 1.5)
density!(ax2, map(x -> x[1, 2]/x[2, 2], res_hg.Σ), color = :transparent, strokecolor = cols[2], strokewidth = 1.5)

Legend(fig[2, 1:2], ax1, orientation = :horizontal)
save("Posterior_Schooling.pdf", fig)


# check the PIPs
pretty_table(
    [mean(res_bric.L, dims = 1)' mean(res_bric.M, dims = 1)'];
    row_labels = col_names,
    header = ["L", "M"]
)
pretty_table(
    [mean(res_hg.L, dims = 1)' mean(res_hg.M, dims = 1)'];
    row_labels = col_names,
    header = ["L", "M"]
)


lines(res_bric.τ[:, 1])
