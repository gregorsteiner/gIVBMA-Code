
using DataFrames, CSV, InvertedIndices, Statistics, StatsPlots, Random

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

# Run analysis
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

Random.seed!(42)
# number of iterations
iters = 100000

# Use rule as main endogenous variable
y = df.lngdpc
x = df.rule
Z = Matrix(df[:, needed_columns[Not([1, 2])]])

res_rule = ivbma(y, x, Z; iter = iters, burn = Int(iters/5))

# malaria as main endogenous variable
x = df.malfal
Z = Matrix(df[:, needed_columns[Not([1, 3])]])

res_malfal = ivbma(y, x, Z; iter = iters, burn = Int(iters/5), dist = "BL")


# Create table summarising the results
function create_latex_table(res_rule, res_malfal)
    row_names = ["maleco", "lnmort", "frost", "humid", "latitude", "eurfrac", "engfrac", "coast", "trade"]

    # Unpack tau estimates for rule and malfal models
    post_mean_and_ci(x) = round.((mean(x), quantile(x, 0.025), quantile(x, 0.975)), digits=2)
    rule_rule_estimate, rule_rule_lower, rule_rule_upper = post_mean_and_ci(res_rule.τ)
    rule_malfal_estimate, rule_malfal_lower, rule_malfal_upper = post_mean_and_ci(res_rule.β[:, 1])
    malfal_rule_estimate, malfal_rule_lower, malfal_rule_upper = post_mean_and_ci(res_malfal.β[:, 1])
    malfal_malfal_estimate, malfal_malfal_lower, malfal_malfal_upper = post_mean_and_ci(res_malfal.τ)

    # \sigma_{12} estimates
    s12_rule_estimate, s12_rule_lower, s12_rule_upper = post_mean_and_ci(map(x -> x[1,2], res_rule.Σ))
    s12_malfal_estimate, s12_malfal_lower, s12_malfal_upper = post_mean_and_ci(map(x -> x[1,2], res_malfal.Σ))

    # PIP table
    PIP_tab = [mean(res_rule.L; dims = 1)' mean(res_rule.M; dims = 1)' mean(res_malfal.L; dims = 1)' mean(res_malfal.M; dims = 1)'][Not(1),:]

    # Start LaTeX table
    table = """
    \\begin{table}[h!]
    \\centering
    \\begin{tabular}{lcccc}
    \\toprule
    & \\multicolumn{2}{c}{\\textbf{rule}} & \\multicolumn{2}{c}{\\textbf{malfal}} \\\\
    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
    & Posterior Mean & 95\\% CI & Posterior Mean & 95\\% CI \\\\
    \\midrule
     rule & $(rule_rule_estimate) &  [$(rule_rule_lower), $(rule_rule_upper)]  & $(malfal_rule_estimate) & [$(malfal_rule_lower), $(malfal_rule_upper)] \\\\
     malfal & $(rule_malfal_estimate) &  [$(rule_malfal_lower), $(rule_malfal_upper)]  & $(malfal_malfal_estimate) & [$(malfal_malfal_lower), $(malfal_malfal_upper)] \\\\
     \\sigma_{12} & $(s12_rule_estimate) &  [$(s12_rule_lower), $(s12_rule_upper)]  & $(s12_malfal_estimate) & [$(s12_malfal_lower), $(s12_malfal_upper)] \\\\
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

println(create_latex_table(res_rule, res_malfal))

