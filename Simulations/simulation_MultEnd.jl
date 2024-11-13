

using BSON, Distributions, LinearAlgebra, Infiltrator

include("competing_methods.jl")
include("bma.jl")

using Pkg; Pkg.activate("../../IVBMA")
using IVBMA


logit(x) = exp(x) / (1+exp(x))

function gen_data(n, c, p = 10)
    Z = rand(MvNormal(zeros(p), I), n)'
    ι = ones(n)

    α, τ, β = (0, [-1/2, 1/2], zeros(p))
    Γ, Δ = ([0, 0], [(ones(p) ./ (1:p)) (ones(p) ./ (1:p))])

    u = rand(MvNormal([0, 0, 0], [1 c c; c 1 c; c c 1]), n)'

    Q = ι * Γ' + Z * Δ + u[:, 2:3] 
    μ, r = (logit.(Q[:, 2]), 1)
    B_α, B_β = (μ * r, r * (1 .- μ))
    X_2 = [rand(Beta(B_α[i], B_β[i])) for i in eachindex(μ)]
    X = [Q[:, 1] X_2]

    y = α*ι + X * τ + Z * β + u[:, 1]

    return (y = y, X = X, Z = Z)
end


d = gen_data(100, 0.25)
res = bma(d.y, d.X, d.Z)

density(res.τ)

