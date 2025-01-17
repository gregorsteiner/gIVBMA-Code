
using DataFrames, CSV, Random, Statistics
using CairoMakie, LaTeXStrings, PrettyTables
using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA
include("../Simulations/bma.jl")
include("../Simulations/competing_methods.jl")

##### Load and prepare data #####
d = CSV.read("card.csv", DataFrame, missingstring = "NA")[:, Not(1:2)]
d.agesq = d.age .^ 2
d_par_educ = d[:, Not(["IQ", "KWW"])] # DataFrame with parents' education => more missing values
d_no_par_educ = d[:, Not(["IQ", "KWW", "fatheduc", "motheduc"])] # DataFrame without parents' education

dropmissing!(d_par_educ)
dropmissing!(d_no_par_educ)

# Data without parents' education
y_1 = Vector(d_no_par_educ.lwage)
X_1 = Matrix(d_no_par_educ[:, ["educ", "exper", "expersq"]])
Z_1 = Matrix(d_no_par_educ[:, ["age", "agesq", "nearc2", "nearc4"]])
W_1 = Matrix(d_no_par_educ[:, ["momdad14", "sinmom14", "step14", "black", "south", "smsa", "married",
                               "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]])

# Data with parents' education
y_2 = Vector(d_par_educ.lwage)
X_2 = Matrix(d_par_educ[:, ["educ", "exper", "expersq"]])
Z_2 = Matrix(d_par_educ[:, ["age", "agesq", "nearc2", "nearc4"]])
W_2 = Matrix(d_par_educ[:, ["fatheduc", "motheduc", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married",
                            "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]])


##### Run analysis #####
Random.seed!(42)
iters = 1000
res_hg_1 = givbma(y_1, X_1, Z_1, W_1; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")
res_bric_1 = givbma(y_1, X_1, Z_1, W_1; iter = iters, burn = Int(iters/5), g_prior = "BRIC")
res_bma_1 = bma(y_1, X_1, W_1; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")

res_hg_2 = givbma(y_2, X_2, Z_2, W_2; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")
res_bric_2 = givbma(y_2, X_2, Z_2, W_2; iter = iters, burn = Int(iters/5), g_prior = "BRIC")
res_bma_2 = bma(y_2, X_2, W_2; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")


# Plot with posterior results
cols = Makie.wong_colors()

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"\tau", ylabel = "(a)")
lines!(ax1, rbw(res_hg_1)[1], color = cols[1], label = "gIVBMA (hyper-g/n)")
lines!(ax1, rbw(res_bric_1)[1], color = cols[2], label = "gIVBMA (BRIC)")
lines!(ax1, rbw_bma(res_bma_1)[1], color = cols[3], label = "BMA (hyper-g/n)")

ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx}")
density!(ax2, map(x -> x[1, 2], res_hg_1.Σ), color = :transparent, strokecolor = cols[1], strokewidth = 1.5)
density!(ax2, map(x -> x[1, 2], res_bric_1.Σ), color = :transparent, strokecolor = cols[2], strokewidth = 1.5)

ax3 = Axis(fig[2, 1], xlabel = L"\tau",  ylabel = "(b)")
lines!(ax3, rbw(res_hg_2)[1], color = cols[1], label = "gIVBMA (hyper-g/n)")
lines!(ax3, rbw(res_bric_2)[1], color = cols[2], label = "gIVBMA (BRIC)")
lines!(ax3, rbw_bma(res_bma_2)[1], color = cols[3], label = "BMA (hyper-g/n)")

ax4 = Axis(fig[2, 2], xlabel = L"\sigma_{yx}")
density!(ax4, map(x -> x[1, 2], res_hg_2.Σ), color = :transparent, strokecolor = cols[1], strokewidth = 1.5)
density!(ax4, map(x -> x[1, 2], res_bric_2.Σ), color = :transparent, strokecolor = cols[2], strokewidth = 1.5)

Legend(fig[3, 1:2], ax1, orientation = :horizontal)
save("Posterior_Schooling.pdf", fig)


# check the PIPs
pretty_table(
    [[repeat([missing], 4); mean(res_hg_1.L, dims = 1)'] mean(res_hg_1.M, dims = 1)'];
    header = ["L", "M"],
    row_labels = ["age", "agesq", "nearc2", "nearc4", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married",
                 "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]
)

pretty_table(
    [[repeat([missing], 4); mean(res_bric_1.L, dims = 1)'] mean(res_bric_1.M, dims = 1)'];
    header = ["L", "M"],
    row_labels = ["age", "agesq", "nearc2", "nearc4", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married",
                 "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]
)

pretty_table(
    [[repeat([missing], 4); mean(res_hg_2.L, dims = 1)'] mean(res_hg_2.M, dims = 1)'];
    header = ["L", "M"],
    row_labels = ["age", "agesq", "nearc2", "nearc4", "fatheduc", "motheduc", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married",
                 "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]
)

pretty_table(
    [[repeat([missing], 4); mean(res_bric_2.L, dims = 1)'] mean(res_bric_2.M, dims = 1)'];
    header = ["L", "M"],
    row_labels = ["age", "agesq", "nearc2", "nearc4", "fatheduc", "motheduc", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married",
                 "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]
)
