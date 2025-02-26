
using DataFrames, CSV, InvertedIndices, Statistics, Random
using CairoMakie

##### Load and prepare dataset #####

df = CSV.read("Carstensen_Gundlach.csv", DataFrame, missingstring="-999.999")

# change column names to match paper
rename!(df, :kaufman => "rule", :mfalrisk => "malfal", :exprop2 => "exprop", :lngdpc95 => "lngdpc",
        :frarom => "trade", :lat => "latitude", :landsea => "coast")

# only keep required columns  
needed_columns = ["lngdpc", "rule", "malfal", "maleco", "lnmort", "frost", "humid",
                  "latitude", "eurfrac", "engfrac", "coast", "trade"]
df = df[:, needed_columns]

# drop all observations with missing values in the variables
dropmissing!(df)

##### Fit full gIVBMA models #####

# Run analysis
using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA
include("../Simulations/bma.jl")


Random.seed!(42)
# number of iterations
iters = 10000

# Use rule as main endogenous variable
y = df.lngdpc
X = [df.rule df.malfal]
Z = Matrix(df[:, needed_columns[Not(1:3)]])

res_bric = givbma(y, X, Z; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "Gaussian", "BL"], g_prior = "BRIC")
res_hg = givbma(y, X, Z; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "Gaussian", "BL"], g_prior = "hyper-g/n")
res_bma = bma(y, X, Z; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")

# plot of posteriors
p = Figure()
ax = Axis(p[1, 1], xlabel = "τ", ylabel = "Density")
lines!(ax, rbw(res_bric)[1], label = "gIVBMA (BRIC)")
lines!(ax, rbw(res_bric)[2])
lines!(ax, rbw(res_hg)[1], label = "gIVBMA (hyper-g/n)", color = Makie.wong_colors()[2])
lines!(ax, rbw(res_hg)[2], color = Makie.wong_colors()[2])
lines!(ax, rbw_bma(res_bma)[1], label = "BMA (hyper-g/n)", color = Makie.wong_colors()[3])
lines!(ax, rbw_bma(res_bma)[2], color = Makie.wong_colors()[3])
axislegend(; position = :lt)
p



# Create table summarising the results
function create_latex_table(res_bric, res_hg, res_bma)
    row_names = ["maleco", "lnmort", "frost", "humid", "latitude", "eurfrac", "engfrac", "coast", "trade"]

    # Unpack tau estimates for rule and malfal models
    post_mean_and_ci(x) = round.((mean(x), quantile(x, 0.025), quantile(x, 0.975)), digits=2)
    rbw_bric = rbw(res_bric)
    rule_bric = post_mean_and_ci(rbw_bric[1])
    malfal_bric = post_mean_and_ci(rbw_bric[2])
    Σ_12_bric = post_mean_and_ci(res_bric.Σ[1, 2, :])
    Σ_13_bric = post_mean_and_ci(res_bric.Σ[1, 3, :])

    rbw_hg = rbw(res_hg)
    rule_hg = post_mean_and_ci(rbw_hg[1])
    malfal_hg = post_mean_and_ci(rbw_hg[2])
    Σ_12_hg = post_mean_and_ci(res_hg.Σ[1, 2, :])
    Σ_13_hg = post_mean_and_ci(res_hg.Σ[1, 3, :])

    rbwbma = rbw_bma(res_bma)
    rule_bma = post_mean_and_ci(rbwbma[1])
    malfal_bma = post_mean_and_ci(rbwbma[2])

    # PIP table
    PIP_tab = [mean(res_bric.L; dims = 2) mean(res_bric.M; dims = 2) mean(res_hg.L; dims = 2) mean(res_hg.M; dims = 2) mean(res_bma.L; dims = 1)']

    # Start LaTeX table
    table = """
    \\begin{table}[h]
    \\centering
    \\begin{tabular}{lcccccc}
    \\toprule
    & \\multicolumn{2}{c}{\\textbf{gIVBMA (BRIC)}} & \\multicolumn{2}{c}{\\textbf{gIVBMA (hyper-g/n)}} & \\multicolumn{2}{c}{\\textbf{BMA (hyper-g/n)}}\\\\
    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}
    & Mean & 95\\% CI & Mean & 95\\% CI & Mean & 95\\% CI \\\\
    \\midrule
     rule & $(rule_bric[1]) &  [$(rule_bric[2]), $(rule_bric[3])]  & $(rule_hg[1]) & [$(rule_hg[2]), $(rule_hg[3])] & $(rule_bma[1]) & [$(rule_bma[2]), $(rule_bma[3])] \\\\
     malfal & $(malfal_bric[1]) &  [$(malfal_bric[2]), $(malfal_bric[3])]  & $(malfal_hg[1]) & [$(malfal_hg[2]), $(malfal_hg[3])] & $(malfal_bma[1]) & [$(malfal_bma[2]), $(malfal_bma[3])] \\\\
     \$\\sigma_{12}\$ & $(Σ_12_bric[1]) &  [$(Σ_12_bric[2]), $(Σ_12_bric[3])]  & $(Σ_12_hg[1]) & [$(Σ_12_hg[2]), $(Σ_12_hg[3])] & - & -\\\\
     \$\\sigma_{13}\$ & $(Σ_13_bric[1]) &  [$(Σ_13_bric[2]), $(Σ_13_bric[3])]  & $(Σ_13_hg[1]) & [$(Σ_13_hg[2]), $(Σ_13_hg[3])] & - & -\\\\
    \\midrule
    & PIP L & PIP M & PIP L & PIP M & PIP L & PIP M \\\\
    \\midrule
    """

    # Add rows with data
    for i in 1:9
        row = row_names[i]
        table *= "$row & $(round(PIP_tab[i,1], digits=2)) & $(round(PIP_tab[i,2], digits=2)) & $(round(PIP_tab[i,3], digits=2)) & $(round(PIP_tab[i,4], digits=2)) &  $(round(PIP_tab[i,5], digits=2)) & - \\\\\n"
    end


    # End LaTeX table
    table *= """
    \\bottomrule
    \\end{tabular}
    \\caption{\\textbf{Geography or institutions?} Treatment effect estimates (posterior mean and 95\\% credible interval) and posterior inclusion probabilities (PIP) in outcome (L) and treatment (M) models for rule and malfal as endogenous variables. The algorithm was run for 10,000 iterations (the first 2,000 of which were discarded as burn-in).}
    \\label{tab:CG_results}
    \\end{table}
    """

    return table
end

println(create_latex_table(res_bric, res_hg, res_bma))


##### LPS analysis #####

include("../Simulations/competing_methods.jl")

function loocv(y, X, Z)
    n = length(y)
    
    meths = ["gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "IVBMA (KL)", "BMA (hyper-g/n)", "TSLS"]
    lps_store = zeros(n, length(meths))
    
    for i in 1:n
        # Split the data: i-th observation is the test set
        y_train = vcat(y[1:i-1], y[i+1:end])
        X_train = vcat(X[1:i-1, :], X[i+1:end, :])
        Z_train = vcat(Z[1:i-1, :], Z[i+1:end, :])
        
        y_test, X_test, Z_test = (y[i:i], X[i:i, :], Z[i:i, :])
        
        # Fit the model on the training set
        fit_bric = givbma(y_train, X_train, Z_train; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "BRIC")
        fit_hg = givbma(y_train, X_train, Z_train; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "hyper-g/n")
        fit_bma = bma(y_train, X_train, Z_train; g_prior = "hyper-g/n")
        
        # Compute LPS for the current test observation
        lps_store[i, :] = [
            lps(fit_bric, y_test, X_test, Z_test),
            lps(fit_hg, y_test, X_test, Z_test),
            ivbma_kl(y_train, X_train, Z_train, y_test, X_test, Z_test).lps,
            lps_bma(fit_bma, y_test, X_test, Z_test),
            tsls(y_train, X_train, Z_train, y_test, X_test, Z_test).lps
        ]
    end
    
    # Return the average LPS across all observations
    return round.(lps_store, digits = 3)
end

Random.seed!(42)
res = loocv(y, X, Z)

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
    table *= "\\bottomrule\n\\end{tabular}\n\\caption{The mean LPS calculated across all iterations of leave-one-out cross-validation on the \\cite{carstensen_primacy_2006} dataset, where each iteration uses a single observation as the holdout set and the rest as the training set.}\n\\label{tab:LOOCV_LPS}\n\\end{table}"
    
    return table
end

methods = ["gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "IVBMA (KL)", "BMA (hyper-g/n)", "TSLS"]
latex_table = create_latex_table(res, methods)
println(latex_table)


##### Endogeneity testing procedure #####

# fit the model with rule being exogenous
Random.seed!(42)
res_hg_1_rule = givbma(y, X[:, 2], [X[:, 1] Z]; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "BL"], g_prior = "hyper-g/n")
res_bric_1_rule = givbma(y, X[:, 2], [X[:, 1] Z]; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "BL"], g_prior = "BRIC")

println(
    "Bayes-factor for exogeneity of rule (hyper-g/n and BRIC respectively): " *
    string(round.(exp(res_hg_1_rule.ML_outcome - res_hg.ML_outcome), digits = 3)) * ", " *
    string(round.(exp(res_bric_1_rule.ML_outcome - res_bric.ML_outcome), digits = 3))
)

# fit the model with malfal being exogenous
Random.seed!(42)
res_hg_1_malfal = givbma(y, X[:, 1], [X[:, 2] Z]; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "Gaussian"], g_prior = "hyper-g/n")
res_bric_1_malfal = givbma(y, X[:, 1], [X[:, 2] Z]; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "Gaussian"], g_prior = "BRIC")
println(
    "Bayes-factor for exogeneity of malfal (hyper-g/n and BRIC respectively): " *
    string(round.(exp(res_hg_1_malfal.ML_outcome - res_hg.ML_outcome), digits = 3)) * ", " *
    string(round.(exp(res_bric_1_malfal.ML_outcome - res_bric.ML_outcome), digits = 3))
)


# Now also test the model with only rule being exogenous against the model with no endogenous variables
Random.seed!(42)
res_hg_0 = givbma(y, X[:, Not(1:2)], [X Z]; iter = iters, burn = Int(iters/5), g_prior = "hyper-g/n")
res_bric_0 = givbma(y, X[:, Not(1:2)], [X Z]; iter = iters, burn = Int(iters/5), g_prior = "BRIC")

println(
    "Bayes-factor for exogeneity of rule (hyper-g/n and BRIC respectively): " *
    string(round.(exp(res_hg_0.ML_outcome - res_hg_1_malfal.ML_outcome), digits = 3)) * ", " *
    string(round.(exp(res_bric_0.ML_outcome - res_bric_1_malfal.ML_outcome), digits = 3))
)
