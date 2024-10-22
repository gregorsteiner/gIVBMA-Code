
using BSON, Pkg

Pkg.develop(path="../../IVBMA")
using IVBMA

include("competing_methods.jl")
include("aux_functions.jl")


"""
    In this setup we get a first stage R^2 ≈ 0.1 with c_M = 3/8 and R^2 ≈ 0.01 with c_M = 1/8.
"""

m = 500
c_M = [1/8, 3/8]

res50 = map(c -> sim_func(m, 50; type = "KO2010", c_M = c, τ = 0.1), c_M)
res500 = map(c -> sim_func(m, 500; type = "KO2010", c_M = c, τ = 0.1), c_M)

bson("SimResKO2010.bson", Dict(:n50 => res50, :n500 => res500))
