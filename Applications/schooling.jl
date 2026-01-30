
using DataFrames, CSV, Random, Statistics, LogExpFunctions
using CairoMakie, LaTeXStrings, KernelDensity
using JLD2

# the following line needs to be run when using the gIVBMA package for the first time
# using Pkg; Pkg.add(url="https://github.com/gregorsteiner/gIVBMA.jl.git")
using gIVBMA

include("../Simulations/bma.jl")
include("../Simulations/competing_methods.jl")


# load data
d = CSV.read("card_imputed.csv", DataFrame)

y = d.lwage
x = d.educ
Z = Matrix(d[:, 3:end])

##### Run analysis (fitting the models can take several hours) #####

# Define models as an array of functions
models = [
    (y, x, Z, iters) -> givbma(y, x, Z; iter = iters, burn = Int(iters/10), 
                 g_prior = "hyper-g/n", cov_prior = "IW"),
    (y, x, Z, iters) -> givbma(y, x, Z; iter = iters, burn = Int(iters/10), 
                 g_prior = "hyper-g/n", cov_prior = "Cholesky", ω_a = 0.1),
    (y, x, Z, iters) -> bma(y, x, Z; iter = iters, burn = Int(iters/10), 
              g_prior = "hyper-g/n"),
    (y, x, Z, iters) -> ivbma_kl(y, x, Z, y, x, Z; s = iters, b = Int(iters/10))
]


# Run in parallel
iters_mcmc = 5000
results = Vector{Any}(undef, length(models))
Threads.@threads for i in eachindex(models)
    Random.seed!(42)
    results[i] = models[i](y, x, Z, iters_mcmc)
end


# fit alternative data without missing values
bool = (.!d.fathmiss) .& (.!d.mothmiss) # only include rows where neither parent's education is missing
y_s, x_s = d.lwage[bool], d.educ[bool]
Z_s = Matrix(d[bool, 3:(end-2)])

results_s = Vector{Any}(undef, length(models))
Threads.@threads for i in eachindex(models)
    Random.seed!(42)
    results_s[i] = models[i](y_s, x_s, Z_s, iters_mcmc)
end


jldsave("Schooling_results.jld2"; 
    iw = results[1], 
    chol = results[2], 
    bma = results[3], 
    ivbma = results[4],
    iw_s = results_s[1],
    chol_s = results_s[2], 
    bma_s = results_s[3], 
    ivbma_s = results_s[4]
)


##### Create plots and tables of the results #####

# Load again
res = load("Schooling_results.jld2")


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
ax1 = Axis(fig[1, 1], xlabel = L"\tau", ylabel = L"Posterior density$$")
lines!(ax1, rbw(res["iw"])[1], color = cols[1], linewidth = wdth, label = L"gIVBMA (IW)$$")
lines!(ax1, rbw(res["chol"])[1], color = cols[2], linestyle = :dot, linewidth = wdth, label = L"gIVBMA ($\omega_a = 0.1$)")
lines!(ax1, rbw_bma(res["bma"])[1], color = cols[3], linestyle = :dashdot, linewidth = wdth, label = L"BMA$$")
kde_ivbma_1 = compute_posterior_density(res["ivbma"].τ_full[:, 1])
lines!(ax1, kde_ivbma_1[1], kde_ivbma_1[2], color = cols[4], linestyle = :dashdotdot, linewidth = wdth, label = L"IVBMA$$")
#xlims!(ax1, -0.01, 0.3)

ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax2, res["iw"].Σ[1, 2, :] ./ res["iw"].Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = wdth)
density!(ax2, res["chol"].Σ[1, 2, :] ./ res["chol"].Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = wdth)
density!(ax2, res["ivbma"].Σ[1, 2, :] ./ res["ivbma"].Σ[2, 2, :] , color = :transparent, linestyle = :dashdotdot, strokecolor = cols[4], strokewidth = wdth)
#xlims!(ax2, -0.31, 0.05)

Legend(fig[2, 1:2], ax1, orientation = :horizontal)
fig 
save("Posterior_Schooling.pdf", fig)


# same plot for the smaller sample
fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"\tau", ylabel = L"Posterior density$$")
lines!(ax1, rbw(res["iw_s"])[1], color = cols[1], linewidth = wdth, label = L"gIVBMA (IW)$$")
lines!(ax1, rbw(res["chol_s"])[1], color = cols[2], linestyle = :dot, linewidth = wdth, label = L"gIVBMA ($\omega_a = 0.1$)")
lines!(ax1, rbw_bma(res["bma_s"])[1], color = cols[3], linestyle = :dashdot, linewidth = wdth, label = L"BMA$$")
kde_ivbma_1 = compute_posterior_density(res["ivbma_s"].τ_full[:, 1])
lines!(ax1, kde_ivbma_1[1], kde_ivbma_1[2], color = cols[4], linestyle = :dashdotdot, linewidth = wdth, label = L"IVBMA$$")
#xlims!(ax1, -0.01, 0.3)

ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax2, res["iw_s"].Σ[1, 2, :] ./ res["iw_s"].Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = wdth)
density!(ax2, res["chol_s"].Σ[1, 2, :] ./ res["chol_s"].Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = wdth)
density!(ax2, res["ivbma_s"].Σ[1, 2, :] ./ res["ivbma_s"].Σ[2, 2, :] , color = :transparent, linestyle = :dashdotdot, strokecolor = cols[4], strokewidth = wdth)
#xlims!(ax2, -0.31, 0.05)

Legend(fig[2, 1:2], ax1, orientation = :horizontal)
fig 
save("Posterior_Schooling_small.pdf", fig)



# Create PIP table
function create_pip_table(iw, chol, ivbma, bma)
    tab_hg = [mean(iw.L, dims = 2) mean(iw.M, dims = 2)]
    tab_bric = [mean(chol.L, dims = 2) mean(chol.M, dims = 2)]
    tab_ivbma = [ivbma.L_bar[Not(1)] ivbma.M_bar[1, :]]
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
    latex *= "& \\multicolumn{2}{c}{gIVBMA (IW)} & "
    latex *= "\\multicolumn{2}{c}{gIVBMA (\$\\omega_a = 0.1 \$)} & "
    latex *= "\\multicolumn{2}{c}{IVBMA} & "
    latex *= "\\multicolumn{1}{c}{BMA} \\\\\n"

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
    create_pip_table(res["iw"], res["chol"], res["ivbma"], res["bma"]),
    names(d)[3:end]
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
