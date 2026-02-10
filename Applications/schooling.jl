
using DataFrames, CSV, Random, Statistics, LogExpFunctions
using CairoMakie, LaTeXStrings, KernelDensity, StatsBase
using JLD2

# the following line needs to be run when using the gIVBMA package for the first time
# using Pkg; Pkg.add(url="https://github.com/gregorsteiner/gIVBMA.jl.git")
using gIVBMA

include("../Simulations/bma.jl")
include("../Simulations/competing_methods.jl")


##### Load and prepare data #####
d = CSV.read("card.csv", DataFrame, missingstring = "NA")[:, Not(1)]

d.expersq = d.exper.^2

covs = ["exper", "expersq", "nearc2", "nearc4", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married", "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669", "fatheduc", "motheduc"]
d = d[:, [["lwage", "educ"]; covs]]

# drop missing values in any column other than parental education
filter!(row -> !any(ismissing, row[1:(end-2)]), d)

# impute the missing parental education by their mean
d.fathmiss = ismissing.(d.fatheduc)
d.mothmiss = ismissing.(d.motheduc)

d.fatheduc = float.(d.fatheduc)
d[d.fathmiss, "fatheduc"] .= mean(d[.!d.fathmiss, "fatheduc"])
d.motheduc = float.(d.motheduc)
d[d.mothmiss, "motheduc"] .= mean(d[.!d.mothmiss, "motheduc"])

# define model matrices
y = d.lwage
x = d.educ
Z = Matrix{Float64}(d[:, 3:end])


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
Z_s = Matrix{Float64}(d[bool, 3:(end-2)])

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

# fit tsls for Comparison
tsls(y, x, Z[:, 4], Z[:, Not(3, 4)], y, x, Z[:, Not(3, 4)]) # Card instruments full sample
tsls(y_s, x_s, Z_s[:, 4], Z_s[:, Not(3, 4)], y_s, x_s, Z_s[:, Not(3, 4)]) # Card instruments reduced sample

tsls(y, x, Z[:, [4, 5, 20, 21]], Z[:, Not(3, 4, 5, 20, 21)], y, x, Z[:, Not(3, 4, 5, 20, 21)]) # gIVBMA instruments full sample
tsls(y_s, x_s, Z_s[:, [ 7, 20, 21]], Z_s[:, Not(3, 4, 7, 20, 21)], y_s, x_s, Z_s[:, Not(3, 4, 7, 20, 21)]) # gIVBMA instruments reduced sample

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

function plot_model_comparison(iw, chol, bma, ivbma; wdth = 2)
    cols = Makie.wong_colors()
    ivbma_color = :purple
    fig = Figure()
    
    # --- Panel 1: Tau Posterior ---
    ax1 = Axis(fig[1, 1], xlabel = L"\tau", ylabel = L"Posterior density$$")

    lines!(ax1, rbw(iw)[1], color = cols[1], linewidth = wdth, label = L"gIVBMA (IW)$$")
    lines!(ax1, rbw(chol)[1], color = cols[2], linestyle = :dot, linewidth = wdth, label = L"gIVBMA ($\omega_a = 0.1$)")
    lines!(ax1, rbw_bma(bma)[1], color = cols[3], linestyle = :dashdot, linewidth = wdth, label = L"BMA$$")

    # IVBMA Slab & Spike
    k_iv = compute_posterior_density(ivbma.τ_full[:, 1])
    # The "Slab"
    lines!(ax1, k_iv[1], k_iv[2], color = ivbma_color, linestyle = :dashdotdot, linewidth = wdth, label = L"IVBMA$$")
    # The "Spike" at zero
    lines!(ax1, [0.0, 0.0], [0.0, k_iv[4]], color = ivbma_color, linewidth = wdth, linestyle = :dashdotdot, )
    scatter!(ax1, [0.0], [k_iv[4]], color = ivbma_color, markersize = 8)

    # --- Panel 2: Sigma Ratio ---
    ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
    
    density!(ax2, iw.Σ[1, 2, :] ./ iw.Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = wdth)
    density!(ax2, chol.Σ[1, 2, :] ./ chol.Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = wdth)
    density!(ax2, ivbma.Σ[1, 2, :] ./ ivbma.Σ[2, 2, :], color = :transparent, linestyle = :dashdotdot, strokecolor = ivbma_color, strokewidth = wdth)

    Legend(fig[2, 1:2], ax1, orientation = :horizontal)
    
    return fig
end


save("Posterior_Schooling.pdf", plot_model_comparison(res["iw"], res["chol"], res["bma"], res["ivbma"]))
save("Posterior_Schooling_small.pdf", plot_model_comparison(res["iw_s"], res["chol_s"], res["bma_s"], res["ivbma_s"]))



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

# PIP table for the full dataset
matrix_to_latex(
    create_pip_table(res["iw"], res["chol"], res["ivbma"], res["bma"]),
    names(d)[3:end]
)

# PIP table for the smaller dataset
matrix_to_latex(
    create_pip_table(res["iw_s"], res["chol_s"], res["ivbma_s"], res["bma_s"]),
    names(d)[3:end]
)



# Creates a side-by-side bar plot comparing the relative frequencies of instruments.
function plot_instrument_comparison(N_Z, N_Z_s; labels=[L"IW$$", L"\omega_a = 0.1", L"IVBMA$$"])
    # Initialize Figure
    fig = Figure(size = (1000, 450), fontsize = 20)
    
    titles = [L"Imputed data$$", L"Complete data$$"]
    data_sets = [N_Z, N_Z_s]
    
    # Iterate to create two subplots
    for i in 1:2
        
        current_tuple = data_sets[i]
        # Determine the global max to define the range from 0
        global_max = maximum(maximum.(current_tuple))
        x_range = 0:global_max # Force start at zero
        
        ax = Axis(fig[1, i], 
            title = titles[i], 
            xlabel = L"Number of Instruments$$", 
            ylabel = i == 1 ? L"Posterior Probability$$" : "",
            xticks = x_range,
            xgridvisible = false)
        
        # 2. Plot grouped bars
        width = 0.25 # Width of individual bars
        offsets = [-width, 0, width]
        colors = [:steelblue, :orange, :forestgreen] # Makie-style colors
        
        for (j, vec) in enumerate(current_tuple)
            # Calculate relative frequencies
            counts = countmap(vec)
            freqs = [get(counts, v, 0) / length(vec) for v in x_range]
            
            barplot!(ax, x_range .+ offsets[j], freqs, 
                width = width, 
                color = (colors[j], 0.8), 
                label = labels[j])
        end
        
        # Add legend to the second plot only
        axislegend(ax, position = :rt)
    end
    
    return fig
end

# create a plot comparing the number of instruments selected
N_Z = (extract_instruments(res["iw"].L, res["iw"].M), extract_instruments(res["chol"].L, res["chol"].M), res["ivbma"].N_Z)
N_Z_s = (extract_instruments(res["iw_s"].L, res["iw_s"].M), extract_instruments(res["chol_s"].L, res["chol_s"].M), res["ivbma_s"].N_Z)

save("Schooling_Instruments.pdf", plot_instrument_comparison(N_Z, N_Z_s))


##### LPS Comparison #####
using ProgressBars

function kfold_cv(y, X, Z; k=5, iters = 500)
    n = length(y)
    fold_size = Int(floor(n / k))
    meths = ["gIVBMA (IW)", "gIVBMA (ω_a = 0.1)", "BMA (hyper-g/n)", "IVBMA", "TSLS"]
    lps_store = zeros(k, length(meths))

    # Generate indices for each fold
    indices = collect(1:n)
    fold_indices = [indices[i:min(i+fold_size-1, n)] for i in 1:fold_size:n]

    for fold in ProgressBar(1:k)
        # Define the test and training indices for the current fold
        test_idx = fold_indices[fold]
        train_idx = setdiff(indices, test_idx)

        # Split the data
        y_train, X_train, Z_train = y[train_idx], X[train_idx], Z[train_idx, :]
        y_test, X_test, Z_test = y[test_idx], X[test_idx], Z[test_idx, :]

        # Fit the model on the training set
        fit_hg = givbma(y_train, X_train, Z_train; g_prior = "hyper-g/n", cov_prior = "IW", iter = iters, burn = Int(iters/5))
        fit_bric = givbma(y_train, X_train, Z_train; g_prior = "hyper-g/n", cov_prior = "Cholesky", ω_a = 0.1, iter = iters, burn = Int(iters/5))
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
res_lps = kfold_cv(y, x, Z)
res_lps_s = kfold_cv(y_s, x_s, Z_s)

function create_latex_table(res1, res2, methods)
    # Calculate means for all three result sets
    means1 = round.(mean(res1, dims=1)[:], digits = 3)
    means2 = round.(mean(res2, dims=1)[:], digits = 3)
    
    # Find the indices of the lowest mean values for each column
    min_indices1 = findall(x -> x == minimum(means1), means1)
    min_indices2 = findall(x -> x == minimum(means2), means2)
    
    # Start building the LaTeX table with booktabs style
    table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcc}\n"
    table *= "\\toprule\n"
    table *= "Method & Imputed (\$n = 3,003\$) & Complete (\$n=2,215\$) \\\\\n"
    table *= "\\midrule\n"
    
    # Fill in the table rows
    for i in eachindex(methods)
        mean_std1 = i in min_indices1 ? "\\textbf{" * string(means1[i]) * "}" : string(means1[i])
        mean_std2 = i in min_indices2 ? "\\textbf{" * string(means2[i]) * "}" : string(means2[i])
        
        table *= methods[i] * " & " * mean_std1 * " & " * mean_std2 * " \\\\\n"
    end
    
    # Close the table
    table *= "\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Returns to schooling:} The mean LPS calculated over each fold of the imputed data (\$n = 3,003\$) and the complete parental education data (\$n = 2,215\$) in a 5-fold cross-validation procedure.}\n\\label{tab:schooling_5_fold_LPS}\n\\end{table}"
    
    return table
end

meths = ["gIVBMA (IW)", "gIVBMA (\$\\omega_a = 0.1 \$)", "BMA", "IVBMA", "TSLS"]
latex_table = create_latex_table(res_lps, res_lps_s, meths)
println(latex_table)
