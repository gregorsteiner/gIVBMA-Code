
using DataFrames, CSV, InvertedIndices, Statistics, StatsPlots, Random

"""
    Load and prepare dataset.
"""

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

"""
    Fit full IVBMA models.
"""

# Run analysis
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

Random.seed!(42)
# number of iterations
iters = 100000

# Use rule as main endogenous variable
y = df.lngdpc
X = [df.rule df.malfal]
Z = Matrix(df[:, needed_columns[Not(1:3)]])

res_bric = ivbma(y, X, Z; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "Gaussian", "BL"], g_prior = "BRIC")
res_hg = ivbma(y, X, Z; iter = iters, burn = Int(iters/5), dist = ["Gaussian", "Gaussian", "BL"], g_prior = "hyper-g/n")

# Create table summarising the results
function create_latex_table(res_bric, res_hg)
    row_names = ["maleco", "lnmort", "frost", "humid", "latitude", "eurfrac", "engfrac", "coast", "trade"]

    # Unpack tau estimates for rule and malfal models
    post_mean_and_ci(x) = round.((mean(x), quantile(x, 0.025), quantile(x, 0.975)), digits=2)
    rbw_bric = rbw(res_bric)
    rule_bric = post_mean_and_ci(rbw_bric[1])
    malfal_bric = post_mean_and_ci(rbw_bric[2])
    Σ_12_bric = post_mean_and_ci(map(x -> x[1,2], res_bric.Σ))
    Σ_13_bric = post_mean_and_ci(map(x -> x[1,3], res_bric.Σ))

    rbw_hg = rbw(res_hg)
    rule_hg = post_mean_and_ci(rbw_hg[1])
    malfal_hg = post_mean_and_ci(rbw_hg[2])
    Σ_12_hg = post_mean_and_ci(map(x -> x[1,2], res_hg.Σ))
    Σ_13_hg = post_mean_and_ci(map(x -> x[1,3], res_hg.Σ))

    # PIP table
    PIP_tab = [mean(res_bric.L; dims = 1)' mean(res_bric.M; dims = 1)' mean(res_hg.L; dims = 1)' mean(res_hg.M; dims = 1)']

    # Start LaTeX table
    table = """
    \\begin{table}[h!]
    \\centering
    \\begin{tabular}{lcccc}
    \\toprule
    & \\multicolumn{2}{c}{\\textbf{BRIC}} & \\multicolumn{2}{c}{\\textbf{hyper-g/n(a=3)}} \\\\
    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
    & Mean & 95\\% CI & Mean & 95\\% CI \\\\
    \\midrule
     rule & $(rule_bric[1]) &  [$(rule_bric[2]), $(rule_bric[3])]  & $(rule_hg[1]) & [$(rule_hg[2]), $(rule_hg[3])] \\\\
     malfal & $(malfal_bric[1]) &  [$(malfal_bric[2]), $(malfal_bric[3])]  & $(malfal_hg[1]) & [$(malfal_hg[2]), $(malfal_hg[3])] \\\\
     \\sigma_{12} & $(Σ_12_bric[1]) &  [$(Σ_12_bric[2]), $(Σ_12_bric[3])]  & $(Σ_12_hg[1]) & [$(Σ_12_hg[2]), $(Σ_12_hg[3])] \\\\
     \\sigma_{13} & $(Σ_13_bric[1]) &  [$(Σ_13_bric[2]), $(Σ_13_bric[3])]  & $(Σ_13_hg[1]) & [$(Σ_13_hg[2]), $(Σ_13_hg[3])] \\\\
    \\midrule
    & PIP L & PIP M & PIP L & PIP M \\\\
    \\midrule
    """

    # Add rows with data
    for i in 1:9
        row = row_names[i]
        table *= "$row & $(round(PIP_tab[i,1], digits=2)) & $(round(PIP_tab[i,2], digits=2)) & $(round(PIP_tab[i,3], digits=2)) & $(round(PIP_tab[i,4], digits=2)) \\\\\n"
    end


    # End LaTeX table
    table *= """
    \\bottomrule
    \\end{tabular}
    \\caption{Treatment effect estimates (posterior mean and 95\\% CI) and posterior inclusion probabilities (PIP) in outcome (L) and treatment (M) models for rule and malfal as endogenous variables. The algorithm was run for 100,000 iterations (the first 20,000 of which were discarded as burn-in).}
    \\label{tab:CG_results}
    \\end{table}
    """

    return table
end

println(create_latex_table(res_bric, res_hg))


"""
    LPS analysis.
"""

include("../Simulations/competing_methods.jl")

function loocv(y, X, Z)
    n = length(y)
    
    meths = ["gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "IVBMA (KL)", "TSLS"]
    lps_store = zeros(n, length(meths))
    
    for i in 1:n
        # Split the data: i-th observation is the test set
        y_train = vcat(y[1:i-1], y[i+1:end])
        X_train = vcat(X[1:i-1, :], X[i+1:end, :])
        Z_train = vcat(Z[1:i-1, :], Z[i+1:end, :])
        
        y_test, X_test, Z_test = (y[i:i], X[i:i, :], Z[i:i, :])
        
        # Fit the model on the training set
        fit_bric = ivbma(y_train, X_train, Z_train; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "BRIC")
        fit_hg = ivbma(y_train, X_train, Z_train; dist = ["Gaussian", "Gaussian", "BL"], g_prior = "hyper-g/n")
        
        # Compute LPS for the current test observation
        lps_store[i, :] = [
            lps(fit_bric, y_test, X_test, Z_test),
            lps(fit_hg, y_test, X_test, Z_test),
            ivbma_kl(y_train, X_train, Z_train, y_test, X_test, Z_test).lps,
            tsls(y_train, X_train, Z_train, y_test, X_test, Z_test).lps
        ]
    end
    
    # Return the average LPS across all observations
    return round.(lps_store, digits = 3)
end

res = loocv(y, X, Z)
boxplot(res, label = permutedims(["gIVBMA (BRIC)", "gIVBMA (hyper-g/n)", "IVBMA (KL)", "TSLS"]))
mean(res, dims = 1)
