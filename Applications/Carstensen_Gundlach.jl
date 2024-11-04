
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

# check Gaussiantiy of endogenous variables
density([df.rule df.malfal])



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

res_malfal = ivbma(y, x, Z; iter = iters, burn = Int(iters/5))



# Create tables
function create_latex_table(data, tau_rule, tau_malfal)
    row_names = ["maleco", "lnmort", "frost", "humid", "latitude", "eurfrac", "engfrac", "coast", "trade"]

    # Unpack tau estimates for rule and malfal models
    tau_rule_estimate, tau_rule_lower, tau_rule_upper = round.(tau_rule, digits=2)
    tau_malfal_estimate, tau_malfal_lower, tau_malfal_upper = round.(tau_malfal, digits=2)

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
     \\tau & $(tau_rule_estimate) &  [$(tau_rule_lower), $(tau_rule_upper)]  & $(tau_malfal_estimate) & [$(tau_malfal_lower), $(tau_malfal_upper)] \\\\
    \\midrule
    & PIP L & PIP M & PIP L & PIP M \\\\
    \\midrule
    """

    # Add rows with data
    for i in 1:9
        row = row_names[i]
        table *= "$row & $(round(data[i,1], digits=2)) & $(round(data[i,2], digits=2)) & $(round(data[i,3], digits=2)) & $(round(data[i,4], digits=2)) \\\\\n"
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


PIP_tab = [mean(res_rule.L; dims = 1)' mean(res_rule.M; dims = 1)' mean(res_malfal.L; dims = 1)' mean(res_malfal.M; dims = 1)'][Not(1),:]
tau_rule = (mean(res_rule.τ), quantile(res_rule.τ, 0.025), quantile(res_rule.τ, 0.975))
tau_malfal = (mean(res_malfal.τ), quantile(res_malfal.τ, 0.025), quantile(res_malfal.τ, 0.975))

println(create_latex_table(PIP_tab, tau_rule, tau_malfal))