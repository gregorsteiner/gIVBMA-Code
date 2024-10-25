
using InvertedIndices
using StatsPlots
using Pkg; Pkg.activate("../../IVBMA")
using IVBMA

include("competing_methods.jl")
include("bma.jl")
include("aux_functions.jl")


τ_true = 1
d = gen_data_KO2010(100, 1/2, τ_true)

res = ivbma(d.y, d.x, d.Z, d.W; iter = 5000)
res_bma = bma(d.y, d.x, d.Z, d.W, 5000)

p = density(
    res.τ, fill = true, alpha = 0.8,
    xlabel = "Causal Effect", ylabel = "Posterior Density", label = "IVBMA"
)
density!(res_bma.τ, fill = true, alpha = 0.8, label = "BMA")
vline!([τ_true], lw = 2, linestyle = :dash, label = "true value")

savefig(p, "Illustration.pdf")
