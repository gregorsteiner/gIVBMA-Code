
using DataFrames, CSV, Random, Statistics, LogExpFunctions
using CairoMakie, LaTeXStrings, KernelDensity, JLD2
using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA
include("../Simulations/bma.jl")
include("../Simulations/competing_methods.jl")

##### Load and prepare data #####
d = CSV.read("card.csv", DataFrame, missingstring = "NA")[:, Not(1)]

d.agesq = d.age .^ 2

covs_1 = ["age", "agesq", "nearc2", "nearc4", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married", "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]
covs_2 = ["age", "agesq", "nearc2", "nearc4", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married", "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669", "fatheduc", "motheduc"]

d_no_par_educ = d[:, [["id", "lwage", "educ"]; covs_1]]
d_par_educ = d[:, [["id", "lwage", "educ"]; covs_2]]

dropmissing!(d_par_educ)
dropmissing!(d_no_par_educ)

# Data without parents' education
y_1 = Vector(d_no_par_educ.lwage)
X_1 = Matrix(d_no_par_educ[:, ["educ"]])
Z_1 = Matrix(d_no_par_educ[:, covs_1])

# Data with parents' education
y_2 = Vector(d_par_educ.lwage)
X_2 = Matrix(d_par_educ[:, ["educ"]])
Z_2 = Matrix(d_par_educ[:, covs_2])


##### Run analysis (fitting the models takes >10hours) #####
Random.seed!(42)
iters = 50000
res_hg_1 = givbma(y_1, X_1, Z_1; iter = iters, burn = Int(iters/10), g_prior = "hyper-g/n")
res_bric_1 = givbma(y_1, X_1, Z_1; iter = iters, burn = Int(iters/10), g_prior = "BRIC")
res_bma_1 = bma(y_1, X_1, Z_1; iter = iters, burn = Int(iters/10), g_prior = "hyper-g/n")
res_ivbma_1 = ivbma_kl(y_1, X_1, Z_1, y_1, X_1, Z_1; s = iters, b = Int(iters/10))

Random.seed!(42)
res_hg_2 = givbma(y_2, X_2, Z_2; iter = iters, burn = Int(iters/10), g_prior = "hyper-g/n")
res_bric_2 = givbma(y_2, X_2, Z_2; iter = iters, burn = Int(iters/10), g_prior = "BRIC")
res_bma_2 = bma(y_2, X_2, Z_2; iter = iters, burn = Int(iters/10), g_prior = "hyper-g/n")
res_ivbma_2 = ivbma_kl(y_2, X_2, Z_2, y_2, X_2, Z_2; s = iters, b = Int(iters/10))

# fit model without parental education on the smaller set of observations
bool = map(x -> x in d_par_educ.id, d_no_par_educ.id)
res_hg_1_small = givbma(y_1[bool], X_1[bool, :], Z_1[bool, :]; iter = iters, burn = Int(iters/10), g_prior = "hyper-g/n")
res_bric_1_small = givbma(y_1[bool], X_1[bool, :], Z_1[bool, :]; iter = iters, burn = Int(iters/10), g_prior = "BRIC")
res_bma_1_small = bma(y_1[bool], X_1[bool, :], Z_1[bool, :]; iter = iters, burn = Int(iters/10), g_prior = "hyper-g/n")
res_ivbma_1_small = ivbma_kl(y_1[bool], X_1[bool, :], Z_1[bool, :], y_1[bool], X_1[bool, :], Z_1[bool, :]; s = iters, b = Int(iters/10))

# save posteriors for later use
jldsave("Posterior_Samples_Schooling.jld2"; res_hg_1, res_bric_1, res_bma_1, res_ivbma_1, res_hg_2, res_bric_2, res_bma_2, res_ivbma_2, res_hg_1_small, res_bric_1_small, res_bma_1_small, res_ivbma_1_small)


##### Create plots and tables of the results #####

# Load again
res = load("Posterior_Samples_Schooling.jld2")
res_hg_1, res_bric_1, res_bma_1, res_ivbma_1 = (res[:"res_hg_1"], res[:"res_bric_1"], res[:"res_bma_1"], res[:"res_ivbma_1"])
res_hg_2, res_bric_2, res_bma_2, res_ivbma_2 = (res[:"res_hg_2"], res[:"res_bric_2"], res[:"res_bma_2"], res[:"res_ivbma_2"])
res_hg_1_small, res_bric_1_small, res_bma_1_small, res_ivbma_1_small = (res[:"res_hg_1_small"], res[:"res_bric_1_small"], res[:"res_bma_1_small"], res[:"res_ivbma_1_small"])


# Plot the posterior results
cols = Makie.wong_colors()
function compute_posterior_density(sample)
    nonzeros = sample[sample .!= 0.0]
    n_total = length(sample)
    prop_nonzero = length(nonzeros) / n_total
    
    kde_obj = kde(nonzeros)
    scaled_density = kde_obj.density * prop_nonzero
    height_point_mass = maximum(kde_obj.density) * (1 - prop_nonzero)

    return (
        kde_obj.x,
        scaled_density,
        1 - prop_nonzero,
        height_point_mass
    )
end

wdth = 2 #line-/strokewidth

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"\tau", ylabel = "(a)")
lines!(ax1, rbw(res_bric_1)[1], color = cols[1], linewidth = wdth, label = "gIVBMA (BRIC)")
lines!(ax1, rbw(res_hg_1)[1], color = cols[2], linestyle = :dot, linewidth = wdth, label = "gIVBMA (hyper-g/n)")
lines!(ax1, rbw_bma(res_bma_1)[1], color = cols[3], linestyle = :dashdot, linewidth = wdth, label = "BMA (hyper-g/n)")
kde_ivbma_1 = compute_posterior_density(res_ivbma_1.τ_full[:, 1])
lines!(ax1, kde_ivbma_1[1], kde_ivbma_1[2], color = cols[4], linestyle = :dashdotdot, linewidth = wdth, label = "IVBMA")
xlims!(ax1, -0.01, 0.3)

ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax2, res_bric_1.Σ[1, 2, :] ./ res_hg_1.Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = wdth)
density!(ax2, res_hg_1.Σ[1, 2, :] ./ res_bric_1.Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = wdth)
density!(ax2, res_ivbma_1.Σ[1, 2, :] ./ res_ivbma_1.Σ[2, 2, :] , color = :transparent, linestyle = :dashdotdot, strokecolor = cols[4], strokewidth = wdth)
xlims!(ax2, -0.31, 0.05)

ax3 = Axis(fig[2, 1], xlabel = L"\tau",  ylabel = "(b)")
lines!(ax3, rbw(res_bric_2)[1], color = cols[1], linewidth = wdth)
lines!(ax3, rbw(res_hg_2)[1], color = cols[2], linestyle = :dot, linewidth = wdth)
lines!(ax3, rbw_bma(res_bma_2)[1], color = cols[3], linestyle = :dashdot, linewidth = wdth)
kde_ivbma_2 = compute_posterior_density(res_ivbma_2.τ_full[:, 1])
lines!(ax3, kde_ivbma_2[1], kde_ivbma_2[2], color = cols[4], linestyle = :dashdotdot, linewidth = wdth)
lines!(ax3, [0.0, 0.0], [0.0, 12.0], color = Makie.wong_colors()[4], linestyle = :dashdotdot, linewidth = wdth)
xlims!(ax3, -0.01, 0.11)

ax4 = Axis(fig[2, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax4, res_bric_2.Σ[1, 2, :] ./ res_hg_2.Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = wdth)
density!(ax4, res_hg_2.Σ[1, 2, :] ./ res_bric_2.Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = wdth)
density!(ax4, res_ivbma_2.Σ[1, 2, :] ./ res_ivbma_2.Σ[2, 2, :], color = :transparent, linestyle = :dashdotdot, strokecolor = cols[4], strokewidth = wdth)
xlims!(ax4, -0.11, 0.05)

Legend(fig[3, 1:2], ax1, orientation = :horizontal)
save("Posterior_Schooling.pdf", fig)


# posterior results for the model excluding parental education on the smaller shared set of observations
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"\tau", ylabel = "(a)")
lines!(ax1, rbw(res_bric_1_small)[1], linewidth = wdth, color = cols[1], label = "gIVBMA (BRIC)")
lines!(ax1, rbw(res_hg_1_small)[1], linewidth = wdth, color = cols[2], linestyle = :dot, label = "gIVBMA (hyper-g/n)")
lines!(ax1, rbw_bma(res_bma_1_small)[1], linewidth = wdth, color = cols[3], linestyle = :dashdot, label = "BMA (hyper-g/n)")
kde_ivbma_1 = compute_posterior_density(res_ivbma_1_small.τ_full[:, 1])
lines!(ax1, kde_ivbma_1[1], kde_ivbma_1[2], linewidth = wdth, color = cols[4], linestyle = :dashdotdot, label = "IVBMA")
lines!(ax1, [0.0, 0.0], [0.0, 18.0], color = Makie.wong_colors()[4], linewidth = wdth, linestyle = :dashdotdot)

ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax2, res_bric_1_small.Σ[1, 2, :] ./ res_hg_1_small.Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = wdth)
density!(ax2, res_hg_1_small.Σ[1, 2, :] ./ res_bric_1_small.Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = wdth)
density!(ax2, res_ivbma_1_small.Σ[1, 2, :] ./ res_ivbma_1_small.Σ[2, 2, :] , color = :transparent, linestyle = :dashdotdot, strokecolor = cols[4], strokewidth = wdth)

Legend(fig[2, 1:2], ax1, orientation = :horizontal)
save("Posterior_Schooling_small.pdf", fig)



# Check posterior of σ_{xx}
fig = Figure()

ax1 = Axis(fig[1, 1], ylabel = "Posterior Density", title = "gIVBMA (BRIC)")
density!(ax1, res_bric_1.Σ[2, 2, :], label = L"No parent edu. ($n = 3{,}003$)")
density!(ax1, res_bric_2.Σ[2, 2, :], label = L"With parent edu. ($n = 2{,}215$)")
density!(ax1, res_bric_1_small.Σ[2, 2, :], label = L"No parent edu. ($n = 2{,}215$)")

ax2 = Axis(fig[1, 2], ylabel = "", title = "gIVBMA (hyper-g/n)")
density!(ax2, res_hg_1.Σ[2, 2, :], label = L"No parent edu. ($n = 3{,}003$)")
density!(ax2, res_hg_2.Σ[2, 2, :], label = L"With parent edu. ($n = 2{,}215$)")
density!(ax2, res_hg_1_small.Σ[2, 2, :], label = L"No parent edu. ($n = 2{,}215$)")

ax3 = Axis(fig[1, 3], ylabel = "", title = "IVBMA")
density!(ax3, res_ivbma_1.Σ[2, 2, :], label = L"No parent edu. ($n = 3{,}003$)")
density!(ax3, res_ivbma_2.Σ[2, 2, :], label = L"With parent edu. ($n = 2{,}215$)")
density!(ax3, res_ivbma_1_small.Σ[2, 2, :], label = L"No parent edu. ($n = 2{,}215$)")

linkyaxes!(ax1, ax2, ax3)
Label(fig[2, 1:3], L"\sigma_{xx}", valign = :top, halign = :center)

Legend(fig[3, 1:3], ax1, orientation = :horizontal, labelsize = 12)

save("Schooling_treatment_variance.pdf", fig)



# Create PIP table
function create_pip_table(hg, bric, ivbma, bma)
    tab_hg = [mean(hg.L, dims = 2) mean(hg.M, dims = 2)]
    tab_bric = [mean(bric.L, dims = 2) mean(bric.M, dims = 2)]
    tab_ivbma = [ivbma.L_bar[Not(1:2)] ivbma.M_bar[1, :]]
    tab_bma = mean(bma.L, dims = 1)'
    return [tab_hg tab_bric tab_ivbma tab_bma]
end


function matrix_to_latex(matrix, rownames)
    # Validate inputs
    num_rows = size(matrix, 1)
    
    # Start building the LaTeX table
    latex = "\\begin{table}[h]\n\\centering\n"

    latex *= "\\begin{tabular}{l"
    # Add column specifications (one for each data column)
    latex *= "c" ^ 7
    latex *= "}\n\\toprule\n"

    # Add multicolumn headers
    latex *= "& \\multicolumn{2}{c}{gIVBMA (hyper-g/n)} & "
    latex *= "\\multicolumn{2}{c}{gIVBMA (BRIC)} & "
    latex *= "\\multicolumn{2}{c}{IVBMA} & "
    latex *= "\\multicolumn{1}{c}{BMA (hyper-g/n)} \\\\\n"

    # Add subheaders
    latex *= "& L & M & L & M & L & M & L \\\\\n"
    latex *= "\\midrule\n"

    # Add data rows
    for i in 1:num_rows
        # Start with rowname
        latex *= rownames[i]
        
        # Add each value in the row
        for j in 1:7
            latex *= " & "
            # Check for missing values
            if ismissing(matrix[i,j])
                latex *= "-"
            else
                # Format number with 3 decimal places using round
                latex *= string(round(matrix[i,j], digits=3))
            end
        end
        latex *= " \\\\\n"
    end

    # Close table and add caption at bottom
    latex *= "\\bottomrule\n\\end{tabular}\n"
    latex *= "\\caption{\\textbf{Returns to schooling:} Posterior inclusion probabilities for the \\cite{card1995collegeproximity} example.}\n"
    latex *= "\\end{table}"

    return println(latex)
end

matrix_to_latex(
    create_pip_table(res_hg_1, res_bric_1, res_ivbma_1, res_bma_1),
    covs_1
)

matrix_to_latex(
    create_pip_table(res_hg_2, res_bric_2, res_ivbma_2, res_bma_2),
    covs_2
)


matrix_to_latex(
    create_pip_table(res_hg_1_small, res_bric_1_small, res_ivbma_1_small, res_bma_1_small),
    covs_1
)

##### LPS Comparison #####
using ProgressBars

function kfold_cv(y, X, Z; k=5, iters = 500)
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
        y_train, X_train, Z_train = y[train_idx], X[train_idx, :], Z[train_idx, :]
        y_test, X_test, Z_test = y[test_idx], X[test_idx, :], Z[test_idx, :]

        # Fit the model on the training set
        fit_hg = givbma(y_train, X_train, Z_train; g_prior = "hyper-g/n", iter = iters, burn = Int(iters/5))
        fit_bric = givbma(y_train, X_train, Z_train; g_prior = "BRIC", iter = iters, burn = Int(iters/5))
        fit_bma = bma(y_train, X_train, Z_train; g_prior = "hyper-g/n", iter = iters, burn = Int(iters/5))
        fit_ivbma = ivbma_kl(y_train, X_train, Z_train, y_test, X_test, Z_test)

        # Compute LPS for the current test observations
        lps_store[fold, :] = [
            lps(fit_hg, y_test, X_test, Z_test),
            lps(fit_bric, y_test, X_test, Z_test),
            lps_bma(fit_bma, y_test, X_test, Z_test),
            fit_ivbma.lps,
            tsls(y_train, X_train, Z_train, y_test, X_test, Z_test).lps
        ]
    end

    return round.(lps_store, digits = 3)
end

Random.seed!(42)
res1 = kfold_cv(y_1, X_1, Z_1)
res2 = kfold_cv(y_2, X_2, Z_2)
bool = map(x -> x in d_par_educ.id, d_no_par_educ.id)
res1_small = kfold_cv(y_1[bool], X_1[bool, :], Z_1[bool, :])


function create_latex_table(res1, res2, res3, methods)
    # Calculate means for all three result sets
    means1 = round.(mean(res1, dims=1)[:], digits = 3)
    means2 = round.(mean(res2, dims=1)[:], digits = 3)
    means3 = round.(mean(res3, dims=1)[:], digits = 3)
    
    # Find the indices of the lowest mean values for each column
    min_indices1 = findall(x -> x == minimum(means1), means1)
    min_indices2 = findall(x -> x == minimum(means2), means2)
    min_indices3 = findall(x -> x == minimum(means3), means3)
    
    # Start building the LaTeX table with booktabs style
    table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccc}\n"
    table *= "\\toprule\n"
    table *= "Method & (a) & (b) & (c) \\\\\n"
    table *= "\\midrule\n"
    
    # Fill in the table rows
    for i in eachindex(methods)
        mean_std1 = i in min_indices1 ? "\\textbf{" * string(means1[i]) * "}" : string(means1[i])
        mean_std2 = i in min_indices2 ? "\\textbf{" * string(means2[i]) * "}" : string(means2[i])
        mean_std3 = i in min_indices3 ? "\\textbf{" * string(means3[i]) * "}" : string(means3[i])
        
        table *= methods[i] * " & " * mean_std1 * " & " * mean_std2 * " & " * mean_std3 * " \\\\\n"
    end
    
    # Close the table
    table *= "\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Returns to schooling:} The mean LPS calculated over each fold of the \\cite{card1995collegeproximity} data (a) without parental education (\$n = 3,003\$), (b) with parental education (\$n = 2,215\$), and (c) without parental education only using the shared set of observations (\$n = 2,215\$) in a 5-fold cross-validation procedure.}\n\\label{tab:schooling_5_fold_LPS}\n\\end{table}"
    
    return table
end

meths = ["gIVBMA (hyper-g/n)", "gIVBMA (BRIC)", "BMA (hyper-g/n)", "IVBMA", "TSLS"]
latex_table = create_latex_table(res1, res2, res1_small, meths)
println(latex_table)
