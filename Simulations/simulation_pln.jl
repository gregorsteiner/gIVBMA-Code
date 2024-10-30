
using BSON
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

include("aux_functions.jl")
include("competing_methods.jl")


m = 500
iters = 2000
c_M = [3/8, 9/16] # corresponding to first stage R^2 of ~0.1 and ~0.2

res50 = map(c -> sim_func(m, 50; type = "PLN", iter = iters, c_M = c, τ = 0.1), c_M)
res500 = map(c -> sim_func(m, 500; type = "PLN", iter = iters, c_M = c, τ = 0.1), c_M)

bson("SimResPLN.bson", Dict(:n50 => res50, :n500 => res500))
