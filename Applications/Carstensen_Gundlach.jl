
using DataFrames, CSV, InvertedIndices

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

y = df.lngdpc
x = df.rule
Z = Matrix(df[:, needed_columns[Not([1, 2])]])


res = ivbma(y, x, Z)
IVBMA.describe(res; pars = ["τ", "β"])

