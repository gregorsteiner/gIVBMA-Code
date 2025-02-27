
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

# save posteriors for later use
jldsave("Posterior_Samples_Schooling.jld2"; res_hg_1, res_bric_1, res_bma_1, res_ivbma_1, res_hg_2, res_bric_2, res_bma_2, res_ivbma_2, res_hg_1_small)


##### Create plots and tables of the results #####

# Load again
res = load("Posterior_Samples_Schooling.jld2")
res_hg_1, res_bric_1, res_bma_1, res_ivbma_1 = (res[:"res_hg_1"], res[:"res_bric_1"], res[:"res_bma_1"], res[:"res_ivbma_1"])
res_hg_2, res_bric_2, res_bma_2, res_ivbma_2 = (res[:"res_hg_2"], res[:"res_bric_2"], res[:"res_bma_2"], res[:"res_ivbma_2"])
res_hg_1_small = res[:"res_hg_1_small"]


# compare models using the smaller set of observations
println(
    "Bayes factor comparing the models excluding and including parental education on the shared set of observations: " * 
    string(exp(res_hg_1_small.ML_outcome + res_hg_1_small.ML_treatment - res_hg_2.ML_outcome - res_hg_2.ML_treatment))
)

res_hg_1_small.ML_outcome - res_hg_2.ML_outcome + res_hg_1_small.ML_treatment - res_hg_2.ML_treatment
res_hg_1.ML_outcome - res_hg_2.ML_outcome + res_hg_1.ML_treatment - res_hg_2.ML_treatment

# Plot the posterior results
cols = Makie.wong_colors()

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel = L"\tau", ylabel = "(a)")
lines!(ax1, rbw(res_hg_1)[1], color = cols[1], label = "gIVBMA (hyper-g/n)")
lines!(ax1, rbw(res_bric_1)[1], color = cols[2], linestyle = :dot, label = "gIVBMA (BRIC)")
lines!(ax1, rbw_bma(res_bma_1)[1], color = cols[3], linestyle = :dashdot, label = "BMA (hyper-g/n)")
kde_ivbma_1 = kde(res_ivbma_1.τ_full[:, 1])
lines!(ax1, kde_ivbma_1.x, kde_ivbma_1.density, color = cols[4], linestyle = :dashdotdot, label = "IVBMA")

ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax2, res_hg_1.Σ[1, 2, :] ./ res_hg_1.Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = 1.5)
density!(ax2, res_bric_1.Σ[1, 2, :] ./ res_bric_1.Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = 1.5)
density!(ax2, res_ivbma_1.Σ[1, 2, :] ./ res_ivbma_1.Σ[2, 2, :] , color = :transparent, linestyle = :dashdotdot, strokecolor = cols[4], strokewidth = 1.5)

ax3 = Axis(fig[2, 1], xlabel = L"\tau",  ylabel = "(b)")
lines!(ax3, rbw(res_hg_2)[1], color = cols[1], label = "gIVBMA (hyper-g/n)")
lines!(ax3, rbw(res_bric_2)[1], color = cols[2], linestyle = :dot, label = "gIVBMA (BRIC)")
lines!(ax3, rbw_bma(res_bma_2)[1], color = cols[3], linestyle = :dashdot, label = "BMA (hyper-g/n)")
kde_ivbma_2 = kde(res_ivbma_2.τ_full[:, 1])
lines!(ax3, kde_ivbma_2.x, kde_ivbma_2.density, color = cols[4], linestyle = :dashdotdot, label = "IVBMA")

ax4 = Axis(fig[2, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}")
density!(ax4, res_hg_2.Σ[1, 2, :] ./ res_hg_2.Σ[2, 2, :], color = :transparent, strokecolor = cols[1], strokewidth = 1.5)
density!(ax4, res_bric_2.Σ[1, 2, :] ./ res_bric_2.Σ[2, 2, :], color = :transparent, linestyle = :dot, strokecolor = cols[2], strokewidth = 1.5)
density!(ax4, res_ivbma_2.Σ[1, 2, :] ./ res_ivbma_2.Σ[2, 2, :], color = :transparent, linestyle = :dashdotdot, strokecolor = cols[4], strokewidth = 1.5)

Legend(fig[3, 1:2], ax1, orientation = :horizontal)
save("Posterior_Schooling.pdf", fig)


# traceplots
tp = Figure()
ax1 = Axis(tp[1, 1], xlabel = "Iteration", ylabel = L"\tau")
lines!(ax1, res_hg_1.τ[1, :], color = cols[1], label = "gIVBMA (hyper-g/n)", alpha = 0.8)
lines!(ax1, res_bric_1.τ[1, :], color = cols[2], linestyle = :dot, label = "gIVBMA (BRIC)", alpha = 0.8)
lines!(ax1, res_bma_1.τ[:, 1], color = cols[3], linestyle = :dashdot, label = "BMA (hyper-g/n)", alpha = 0.8)
lines!(ax1, res_ivbma_1.τ_full[:, 1], color = cols[4], linestyle = :dashdotdot, label = "IVBMA", alpha = 0.8)

ax2 = Axis(tp[2, 1], xlabel = "Iteration", ylabel = L"\tau")
lines!(ax2, res_hg_2.τ[1, :], color = cols[1], label = "gIVBMA (hyper-g/n)", alpha = 0.8)
lines!(ax2, res_bric_2.τ[1, :], color = cols[2], linestyle = :dot, label = "gIVBMA (BRIC)", alpha = 0.8)
lines!(ax2, res_bma_2.τ[:, 1], color = cols[3], linestyle = :dashdot, label = "BMA (hyper-g/n)", alpha = 0.8)
lines!(ax2, res_ivbma_2.τ_full[:, 1], color = cols[4], linestyle = :dashdotdot, label = "IVBMA", alpha = 0.8)

Legend(tp[3, 1], ax1, orientation = :horizontal)
save("Traceplots_Schooling.pdf", tp)


# check for high probability models
function find_frequent_models(res, k)
   # Create array of tuples containing row combinations
   n_rows = size(res.L, 1)
   models = [(res.L[i,:], res.M[i,:]) for i in 1:n_rows]
   
   # Count frequency of each unique model
   model_counts = Dict{Tuple{Vector{Int}, Vector{Int}}, Int}()
   for model in models
       model_counts[model] = get(model_counts, model, 0) + 1
   end
   
   # Sort models by frequency (descending) and take top k
   sorted_models = sort(collect(model_counts), by=x->x[2], rev=true)[1:min(k, length(model_counts))]
   
   # Calculate probabilities and format output
   total_models = n_rows
   results = []
   
   for (model, count) in sorted_models
       # Convert L and M rows to separate strings
       L_row, M_row = model
       L_string = join(L_row, "")
       M_string = join(M_row, "")
       
       # Calculate probability and round to 3 digits
       probability = round(count/total_models, digits=3)
       
       push!(results, (L=L_string, M=M_string, probability=probability))
   end
   
   return results
end


function generate_latex_table(results_excl_edu, results_incl_edu)
    
    # Method names
    method_names = ["gIVBMA (hyper-g/n)", "gIVBMA (BRIC)", "IVBMA"]
    
    # Start LaTeX table with booktabs style
    table = "\\begin{table}[htbp]\n"
    table *= "\\centering\n"
    table *= "\\caption{Top Models by Method and Scenario}\n"
    table *= "\\label{tab:top_models}\n"

    # Loop through each scenario
    scenarios = ["Excluding parental education", "Including parental education"]
    results_sets = [results_excl_edu, results_incl_edu]
    
    for (s, scenario) in enumerate(scenarios)
        # Subtitle for the scenario
        table *= "\\textbf{$scenario} \\\\\n"
        
        # Loop through each method
        for (m, method) in enumerate(method_names)
            # Get results for this method/scenario
            results = results_sets[s][m]
            num_models =  length(results)
            
            # Header for this method
            table *= "\\textbf{$method} \\\\\n"
            
            # Create a table for this method
            table *= "\\begin{tabular}{l ccc}\n"
            table *= "\\toprule\n"
            table *= "Model & L & M & Probability \\\\\n"
            table *= "\\midrule\n"
            
            # Show up to num_models models
            models_to_show = min(num_models, length(results))
            for j in 1:models_to_show
                model = results[j]
                table *= "$j & $(model.L) & $(model.M) & $(model.probability) \\\\\n"
            end
            
            table *= "\\bottomrule\n"
            table *= "\\end{tabular}\n"
            
            # Add space between methods
            if m < length(method_names) || s < length(scenarios)
                table *= "\n\\vspace{2ex}\n\n"
            end
        end
        
        # Add extra space between scenarios
        if s < length(scenarios)
            table *= "\\vspace{3ex}\n"
        end
    end
    
    # Close table
    table *= "\\end{table}\n"
    
    return table
end

results_excl_edu = [
    find_frequent_models(res_hg_1.L, res_hg_2.M, 5),
    find_frequent_models(res_bric_1.L, res_bric_2.M, 5),
    find_frequent_models(res_ivbma_1.L, res_ivbma_2.M[:, 1, :]', 5)
]

results_incl_edu = [
    find_frequent_models(res_hg_2.L, res_hg_2.M, 5),
    find_frequent_models(res_bric_2.L, res_bric_2.M, 5),
    find_frequent_models(res_ivbma_2.L, res_ivbma_2.M[:, 1, :]', 5)
]

generate_latex_table(results_excl_edu, results_incl_edu) |> println

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
    latex *= "\\caption{\\textbf{Returns to schooling:} Posterior inclusion probabilities for the \\cite{card1995collegeproximity} example. The IVBMA posterior inclusion probabilities are for education.}\n"
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

function create_latex_table(res1, res2, methods)
    # Calculate means for both result sets
    means1 = round.(mean(res1, dims=1)[:], digits = 3)
    means2 = round.(mean(res2, dims=1)[:], digits = 3)
    
    # Find the indices of the lowest mean values for each column
    min_index1 = argmin(means1)
    min_index2 = argmin(means2)
    
    # Start building the LaTeX table with booktabs style
    table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcc}\n"
    table *= "\\toprule\n"
    table *= "Method & Without parental education & With parental education \\\\\n"
    table *= "\\midrule\n"
    
    # Fill in the table rows
    for i in eachindex(methods)
        mean_std1 = string(means1[i])
        mean_std2 = string(means2[i])
        
        # Highlight the minimum values in each column
        if i == min_index1
            mean_std1 = "\\textbf{" * mean_std1 * "}"
        end
        if i == min_index2
            mean_std2 = "\\textbf{" * mean_std2 * "}"
        end
        
        table *= methods[i] * " & " * mean_std1 * " & " * mean_std2 * " \\\\\n"
    end
    
    # Close the table
    table *= "\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Returns to schooling:} The mean LPS calculated over each fold of the \\cite{card1995collegeproximity} data in a 5-fold cross-validation procedure.}\n\\label{tab:schooling_5_fold_LPS}\n\\end{table}"
    
    return table
end

meths = ["gIVBMA (hyper-g/n)", "gIVBMA (BRIC)", "BMA (hyper-g/n)", "IVBMA", "TSLS"]
latex_table = create_latex_table(res1, res2, meths)
println(latex_table)
