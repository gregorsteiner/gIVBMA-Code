using BSON

# Load the simulation results
res = BSON.load("SimResKO2010.bson")

# Helper function to format individual results into a LaTeX tabular format
function format_result(res)
    tab = vcat(res.RMSE, res.Bias, res.Coverage', res.lpd')'
    return round.(tab, digits = 2)
end

# Function to combine tables into a stacked multicolumn table
function make_stacked_multicolumn_table(res)
    # Extract the tables for each scenario
    table_50_001 = format_result(res[:n50][1])
    table_50_01 = format_result(res[:n50][2])
    table_500_001 = format_result(res[:n500][1])
    table_500_01 = format_result(res[:n500][2])

    # Header for each method
    methods = ["IVBMA", "IVBMA-2C", "TSLS", "O-TSLS", "JIVE", "RJIVE", "Post-LASSO", "MATSLS"]

    # Start the LaTeX table (without math mode)
    table_str = "\\begin{table}\n\\centering\n\\begin{tabular}{l*{8}{r}}\n\\toprule\n"

    # First row: RMSE, Bias, Coverage, LPS for n = 50
    table_str *= " & \\multicolumn{8}{c}{n = 50} \\\\\n"
    
    # Second row: showing R_f^2 for each n = 50 case
    table_str *= " & \\multicolumn{4}{c}{R_f^2 = 0.01} & \\multicolumn{4}{c}{R_f^2 = 0.1} \\\\\n"
    
    # Third row: RMSE, Bias, Coverage, LPS labels for each R^2
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Add rows for each method (for n = 50)
    for i in 1:length(methods)
        table_str *= methods[i] * " & "
        table_str *= string(table_50_001[i, 1]) * " & " * string(table_50_001[i, 2]) * " & " * string(table_50_001[i, 3]) * " & " * string(table_50_001[i, 4]) * " & "
        table_str *= string(table_50_01[i, 1]) * " & " * string(table_50_01[i, 2]) * " & " * string(table_50_01[i, 3]) * " & " * string(table_50_01[i, 4]) * " \\\\\n"
    end

    # Add a midrule separator for clarity before starting the n = 500 part
    table_str *= "\\midrule\n"

    # Add another block for n = 500 scenarios
    table_str *= " & \\multicolumn{8}{c}{n = 500} \\\\\n"
    table_str *= " & \\multicolumn{4}{c}{R_f^2 = 0.01} & \\multicolumn{4}{c}{R_f^2 = 0.1} \\\\\n"
    table_str *= "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
    table_str *= " & \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} "
    table_str *= "& \\textbf{RMSE} & \\textbf{Bias} & \\textbf{Cov.} & \\textbf{LPS} \\\\\n\\midrule\n"

    # Add rows for each method (for n = 500)
    for i in 1:length(methods)
        table_str *= methods[i] * " & "
        table_str *= string(table_500_001[i, 1]) * " & " * string(table_500_001[i, 2]) * " & " * string(table_500_001[i, 3]) * " & " * string(table_500_001[i, 4]) * " & "
        table_str *= string(table_500_01[i, 1]) * " & " * string(table_500_01[i, 2]) * " & " * string(table_500_01[i, 3]) * " & " * string(table_500_01[i, 4]) * " \\\\\n"
    end

    # Finish the table
    table_str *= "\\bottomrule\n\\end{tabular}\n"
    table_str *= "\\caption{RMSE, bias, credible (or confidence) interval coverage (nominal 95\\%) and mean LPS (lower is better) on 1,000 simulated datasets. RMSE and Bias are based on the posterior mean of IVBMA.}\n"
    table_str *= "\\label{tab:KO_Sim}\n\\end{table}"

    return table_str
end

# Generate and print the LaTeX table with stacked multicolumns
latex_table = make_stacked_multicolumn_table(res)
println(latex_table)
