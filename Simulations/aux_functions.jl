
using Distributions, LinearAlgebra, GLMNet

"""
    Auxiliary function to generate the instruments' coefficients in the KO2010 design.
    Note that with p = 20 choosing c_M = 3/8 leads to an R^2 of approximately 0.1.
"""
function gen_instr_coeff(p::Integer, c_M::Number)
    res = zeros(p)
    for i in 1:p
        if i <= p/2
            res[i] = c_M * (1 - i/(p/2 + 1))^4
        end
    end
    return res
end

"""
    Generate data for the simulation based on one of the following designs:
        1. Kuersteiner & Okui (2010)
        2. Kang et al (2016)
"""
function gen_data_KO2010(n::Integer = 100, c_M::Number = 3/8, τ::Number = 0.1, p::Integer = 20, k::Integer = 10, c::Number = 1/2)
    V = rand(MvNormal(zeros(p+k), I), n)'
    Z = V[:,1:p]
    W = V[:,(p+1):(p+k)]

    α = 1; γ = 1
    δ_Z = gen_instr_coeff(p, c_M)
    δ_W = [ones(Int(k/2)); zeros(Int(k/2))] .* 0.1
    β = [ones(Int(k/2)); zeros(Int(k/2))]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ_Z + W * δ_W + u[:,2]
    y = α .+ τ * x .+ W * β + u[:,1]

    # centre all regressors
    y = y .- mean(y)
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)
    W = W .- mean(W; dims = 1)

    return (y=y, x=x, Z=Z, W=W)
end

function gen_data_Kang2016(n::Integer = 200, τ::Number = 0.1, p::Integer = 10, s::Integer = 2, c::Number = 0.5)
    Z = rand(MvNormal(zeros(p), I), n)'

    α = γ = 1
    δ = ones(p) .* 5/32 # chosen s.t. the first-staeg R^2 is approximately 0.2
    β = [ones(s); zeros(p-s)]

    u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
    x = γ .+ Z * δ + u[:,2]
    y = α .+ τ * x .+ Z * β + u[:,1]

    # centre all regressors
    y = y .- mean(y)
    x = x .- mean(x)
    Z = Z .- mean(Z; dims = 1)

    return (y=y, x=x, Z=Z)
end


"""
    This function generates the quantites of interest for each method.
"""
function meth_KO2010(n::Integer, c_M::Number, τ::Number, iter::Integer; level = 0.05)
    d = gen_data_KO2010(n, c_M, τ)
    d_h = gen_data_KO2010(Int(n/5), c_M, τ)

    # full posterior samples for each bma variant
    bma_meths = [iv_fit, ivbma, ivbma_2c]
    try
        global res_full = map(f -> f(d.y, d.x, d.Z, d.W; iter = iter, burn = Int(iter/2)), bma_meths)
    catch e
        d = gen_data_KO2010(n, c_M, τ)
        global res_full = map(f -> f(d.y, d.x, d.Z, d.W; iter = iter, burn = Int(iter/2)), bma_meths)
    end

    # compute posterior mean, credible interval and lpd
    res = map(x -> (τ = mean(x.τ), CI = quantile(x.τ, [level/2, 1 - level/2]), lpd = lpd(x, d_h.y, d_h.x, d_h.Z, d_h.W)), res_full)

    res = [
        res;
        tsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W; level = level);
        tsls(d.y, d.x, d.Z[:, 1:10], d.W[:, 1:5], d_h.y, d_h.x, d_h.W[:, 1:5]; level = level);
        jive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W; level = level);
        rjive(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W; level = level);
        post_lasso(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W; level = level);
        matsls(d.y, d.x, d.Z, d.W, d_h.y, d_h.x, d_h.W; level = level)
    ]

    return res
end

function meth_Kang2016(n::Integer, p::Number, s::Number, τ::Number, iter::Integer; level = 0.05)
    d = gen_data_Kang2016(n, τ, p, s)
    d_h = gen_data_Kang2016(Int(n/5), τ, p, s)

    # full posterior samples for each bma variant
    try
        global res_full = ivbma(d.y, d.x, d.Z; iter = iter, burn = Int(iter/2))
    catch
        d = gen_data_Kang2016(n, τ, p, s)
        global res_full = ivbma(d.y, d.x, d.Z; iter = iter, burn = Int(iter/2))
    end

    # compute posterior mean, credible interval and lpd
    res = (
        τ = mean(res_full.τ),
        CI = quantile(res_full.τ, [level/2, 1 - level/2]),
        lpd = lpd(res_full, d_h.y, d_h.x, d_h.Z)
    )
    
    res = [
        res;
        tsls(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z)
        tsls(d.y, d.x, d.Z[:, (s+1):p], d_h.y, d_h.x, d_h.Z[:, (s+1):p])
        sisVIVE(d.y, d.x, d.Z, d_h.y, d_h.x, d_h.Z; level = level)
    ]

    return res
end


"""
    This function performs the simulation and returns the RMSE, coverage and CI width for each method.
"""
function sim_func(m::Integer, n::Integer; c_M::Number = 3/8, τ::Number = 1, iter::Integer = 5000, p = 10, s = 2, level = 0.05, type::String = "KO2010")
    taus = Vector(undef, m)
    covg = Vector(undef, m)
    width = Vector(undef, m)
    lpds = Vector(undef, m)
    
    for i in 1:m
        if type == "KO2010"
            res = meth_KO2010(n, c_M, τ, iter; level = level)
        elseif type == "Kang2016"
            res = meth_Kang2016(n, p, s, τ, iter; level = level)
        end
        taus[i] = map(x -> x.τ, res)
        covg[i] = map(x -> (x.CI[1] < τ) & (x.CI[2] > τ), res)
        width[i] = map(x -> x.CI[2] - x.CI[1], res)
        lpds[i] = map(x -> x.lpd, res)

        println(string(i) * "/" * string(m)) # add progress
    end

    rmse = mapslices(x -> sqrt(mean((x .- τ).^2)), reduce(vcat, taus'); dims = 1)
    bias = mapslices(x -> mean((x .- τ)), reduce(vcat, taus'); dims = 1)
    covg = mean(covg; dims = 1)[1]
    width = mean(width; dims = 1)[1]
    lpd = mean(lpds; dims = 1)[1]

    return (RMSE = rmse, Bias = bias, Coverage = covg, Width = width, lpd = lpd)
end
