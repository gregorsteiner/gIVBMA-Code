
using StatFiles, DataFrames, ShiftedArrays
df = DataFrame(StatFiles.load("rs/gmm_final.dta"))


"""
    Prepare Data
"""

rhsendog = [:yc_penn, :le_wdi, :sw1_i, :linf1, :M2_GDP_ES, :BB_GDP_WDI, :revol]
rhsexog = [:ethfrac, :geog6099, :safrica, :east]
t_vars = [Symbol("tdum" * string(i)) for i in 1:8]  # assuming tdum variables are named tdum1, tdum2, ..., tdum8

zrhs = [:yc_penn, :le_wdi, :sw1_i, :INST_QLTY, :linf1, :M2_GDP_ES, :BB_GDP_WDI, :revol]
aid_gdp = :aid_gdp  # the additional variable in zrhs for instrument creation

# Generate lagged and differenced variables for instruments
for t in 2:8
    for lag in 1:(t - 1)
        for var in vcat(zrhs, aid_gdp)
            # Generate lagged instruments with correct string interpolation
            df[!, Symbol("z$(t)L$(lag)L$(var)")] = ifelse.(df.t .== t, ShiftedArrays.lag(df[!, var], lag), 0)
            
            # Generate first-difference instruments
            df[!, Symbol("z$(t)L$(lag)D$(var)")] = ifelse.(df.t .== t, ShiftedArrays.lag(df[!, var], lag) - ShiftedArrays.lag(df[!, var], lag + 1), 0)
        end
    end
end


# Set missing values to zero for generated instrument variables
for col in names(df)
    if startswith(col, "z")  # only apply to z* variables
        df[ismissing.(df[!, col]), col] .= 0
    end
end

# Generate additional lagged variables for zrhs and aid_gdp (up to 7 lags)
for var in vcat(zrhs, aid_gdp)
    for l in 1:7
        # Create lagged levels
        df[!, Symbol("L$(l)L$(var)")] = Vector(ShiftedArrays.lag(df[!, var], l))
        
        # Create lagged differences
        df[!, Symbol("L$(l)D$(var)")] = Vector(ShiftedArrays.lag(df[!, var], l) - ShiftedArrays.lag(df[!, var], l + 1))
    end
end

# Set missing values to zero for generated L* variables
for col in names(df)
    if startswith(col, "L")  # only apply to L* variables
        df[ismissing.(df[!, col]), col] .= 0
    end
end


instrument_pattern = r"^z"  # All columns starting with `z`
instrument_columns = Symbol.(filter(name -> occursin(instrument_pattern, String(name)), names(df)))
control_vars = vcat(rhsexog, Symbol.([col for col in names(df) if occursin(r"^tdum", String(col))]))  # rhsexog and time dummies
control_vars = control_vars[Not(5)]

all_columns_needed = union([:rgdpchg, :aid_gdp], union(instrument_columns, control_vars))
df = dropmissing(df, all_columns_needed)

Z = Matrix(select(df, instrument_columns))
W = Matrix(select(df, control_vars))

y = df.rgdpchg
x = df.aid_gdp



"""
    Fit models
"""

using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

m = [size(W, 2)/2, size(Z, 2) / 20]
res = ivbma(y, x, Z, W; m = m)

