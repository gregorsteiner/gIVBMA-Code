
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
iters = 5000
res_hg_1 = givbma(y_1, X_1, Z_1, W_1; iter = iters, burn = Int(iters/2), g_prior = "hyper-g/n")
res_bric_1 = givbma(y_1, X_1, Z_1, W_1; iter = iters, burn = Int(iters/2), g_prior = "BRIC")
res_bma_1 = bma(y_1, X_1, W_1; iter = iters, burn = Int(iters/2), g_prior = "hyper-g/n")

res_hg_2 = givbma(y_2, X_2, Z_2, W_2; iter = iters, burn = Int(iters/2), g_prior = "hyper-g/n")
res_bric_2 = givbma(y_2, X_2, Z_2, W_2; iter = iters, burn = Int(iters/2), g_prior = "BRIC")
res_bma_2 = bma(y_2, X_2, W_2; iter = iters, burn = Int(iters/2), g_prior = "hyper-g/n")

# save model objects
using BSON
bson("schooling_models.bson", Dict(
    :HG_A => res_hg_1,
    :BRIC_A => res_bric_1,
    :BMA_A => res_bma_1,
    :HG_B => res_hg_2,
    :BRIC_B => res_bric_2,
    :BMA_B => res_bma_2
))

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


##### LPS Comparison #####
using ProgressBars

function kfold_cv(y, X, Z, W; k=5, iters = 1000)
    n = length(y)
    fold_size = Int(floor(n / k))
    meths = ["gIVBMA (hyper-g/n)", "gIVBMA (BRIC)", "BMA (hyper-g/n)", "IVBMA", "TSLS"]
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
        fit_hg = givbma(y_train, X_train, Z_train, W_train; g_prior = "hyper-g/n", iter = iters, burn = Int(iters/5))
        fit_bric = givbma(y_train, X_train, Z_train, W_train; g_prior = "BRIC", iter = iters, burn = Int(iters/5))
        fit_bma = bma(y_train, X_train, W_train; g_prior = "hyper-g/n", iter = iters, burn = Int(iters/5))

        # Compute LPS for the current test observations
        lps_store[fold, :] = [
            lps(fit_hg, y_test, X_test, Z_test, W_test),
            lps(fit_bric, y_test, X_test, Z_test, W_test),
            lps_bma(fit_bma, y_test, X_test, W_test),
            ivbma_kl(y_train, X_train, Z_train, W_train, y_test, X_test, Z_test, W_test).lps,
            tsls(y_train, X_train, Z_train, W_train, y_test, X_test, W_test).lps
        ]
    end

    return round.(lps_store, digits = 3)
end

Random.seed!(42)
res = kfold_cv(y_2, X_2, Z_2, W_2)

function create_latex_table(res, methods)
    # Calculate means and standard deviations
    means = round.(mean(res, dims=1)[:], digits = 3)
    
    # Find the index of the lowest mean value
    min_index = argmin(means)
    
    # Start building the LaTeX table with booktabs style
    table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lc}\n"
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
    table *= "\\bottomrule\n\\end{tabular}\n\\caption{The mean LPS calculated over each fold of the \\cite{card1995collegeproximity} data in a 5-fold cross-validation procedure.}\n\\label{tab:schooling_5_fold_LPS}\n\\end{table}"
    
    return table
end

meths = ["gIVBMA (hyper-g/n)", "gIVBMA (BRIC)", "BMA (hyper-g/n)", "IVBMA", "TSLS"]
latex_table = create_latex_table(res, meths)
println(latex_table)
