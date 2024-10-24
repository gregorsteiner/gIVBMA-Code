
using BSON

using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

include("competing_methods.jl")
include("sisVIVE.jl")
include("aux_functions.jl")


m = 200
ss = [3, 6]

res50 = map(s -> sim_func(m, 50; type = "Kang2016", s = s), ss)
res500 = map(s -> sim_func(m, 500; type = "Kang2016", s = s), ss)

bson("SimResKang2016.bson", Dict(:n50 => res50, :n500 => res500))



