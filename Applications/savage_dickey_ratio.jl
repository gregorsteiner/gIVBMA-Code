
##### This file contains functions to compute the Savage-Dickey ratio based on a gIVBMA sample #####

# function that returns the Rao-Blackwellised distribution for the off-diagonal given a vector of covariance matrices and scale matrices
function rb_Σ_yx(Σ_vec, S_vec; k = 1)    
    pars = map((Σ, S) -> begin
        σ_yy = Σ[1, 1]
        Σ_22_1 = Σ[2:end, 2:end] - Σ[2:end, 1] * Σ[2:end, 1]' / σ_yy
        s_yy, S_yx = (S[1, 1], S[2:end, 1])
        return (σ_yy/s_yy * S_yx, σ_yy^2/s_yy * Σ_22_1) 
    end, Σ_vec, S_vec)
    dist = MixtureModel(map(p -> Normal(p[1][k], sqrt(p[2][k, k])), pars))
end

# function that computes the posterior scale matrix given a gIVBMA object
function post_scale(res)
    n = length(res.y)
    S_store = Vector{Matrix{Float64}}(undef, length(res.Σ))
    W_c = res.W .- mean(res.W; dims = 1)
    
    for i in eachindex(res.α)
        ϵ = res.Q[i, :, 1] - ones(n) * res.α[i] - res.X * res.τ[i, :] - W_c * res.β[i, :]
        H = res.Q[i, :, 2:end] - ones(n) * res.Γ[i, :]' - W_c * res.Δ[i, :, :]
        S_store[i] = [ϵ H]' * [ϵ H] + I
    end
    return S_store
end

# compute the SD ratio
function sd_ratio(res; k = 1)
    l = size(res.X, 2)
    
    # get prior distribution of Σ_{yx}
    m = 100000
    ν = rand(Exponential(1), m) .+ (l+1)
    Σ_prior = map(x -> rand(InverseWishart(x, Matrix{Float64}(I, l+1, l+1))), ν)
    S_prior = [Matrix{Float64}(I, 3, 3) for _ in 1:m] # the prior scale matrix
    d_prior = rb_Σ_yx(Σ_prior, S_prior; k = k)

    # get posterior distribution of Σ_{yx}
    Σ_post = res.Σ
    S_post = post_scale(res)
    d_post = rb_Σ_yx(Σ_post, S_post; k = k)

    # return ratio
    sd_ratio = pdf(d_post, 0) / pdf(d_prior, 0)
    return sd_ratio
end