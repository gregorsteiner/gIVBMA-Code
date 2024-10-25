using SpecialFunctions, Distributions, Statistics

"""
    This file implements a 'naive' BMA approach, i.e. only on the outcome model ignoring the treatment model.
    This corresponds to ignoring the endogeneity.
"""

function post_sample(y::Vector, x::Vector, W::Matrix, g::Number)
    n = length(y)
    
    y_bar = Statistics.mean(y)
    U = [x W]
    δ = g / (g+1)
    U_i = [ones(n) U]

    Q = I - U_i * inv(U_i'U_i) * U_i'

    s = δ * y' * Q * y + (1-δ) * (y .- y_bar)' * (y .- y_bar)
    σ = sqrt(rand(InverseGamma((n-1)/2, s/2))) 
    α = rand(Normal(y_bar, σ/sqrt(n))) 
    β_t = rand(MvNormal(δ * inv(U'U)*U'y, Symmetric(σ^2 * δ * inv(U'U))))
    τ = β_t[1]
    β = β_t[2:end]
    
    return (α = α, τ = τ, β = β, σ = σ)
end

function marginal_likelihood(y::Vector, W::Matrix, g::Number)
    n = size(W, 1)
    k_j = size(W, 2)
    
    X_i = [ones(n) W]
    Q = I - X_i*inv(X_i'X_i)*X_i'
    
    invDetCoef = y'Q * y / dot((y .- mean(y)), (y .- mean(y)))
    
    res = ((n-1-k_j)/2) * log((1+g)) + (-(n-1)/2) * log(1 + g*(invDetCoef))
    return res
end

function model_prior(x::Vector, k::Integer; a::Number = 1, m::Number = floor(k/2))
    b = (k - m) / m 
    kj = sum(x)
    
    Γ(x) = SpecialFunctions.gamma(x)
    res = log(Γ(a+b)) - log(Γ(a) * Γ(b)) + log(Γ(a+kj) * Γ(b+k-kj)) - log(Γ(a+b+k))
    return res
end

function bma(y::AbstractVector, x::AbstractVector, Z::AbstractMatrix, W::AbstractMatrix, iter::Integer = 2000, burn::Integer = 1000)

    n = size(W, 1)
    k = size(W, 2)
    g = max(n, k^2)
    m = floor(k/2)

    x = x .- mean(x)
    W = W .- mean(W; dims = 1)


    α_store = zeros(iter)
    τ_store = zeros(iter)
    β_store = zeros(iter, k)
    σ_store = zeros(iter)

    incl = Array{Bool}(undef, (iter), k)
    incl[1,:] = repeat([true], k)


    for i in 2:(iter)
        # draw proposal
        ind = sample(1:k) # draw index to permute
        Curr = incl[i-1, :] # get current inclusion
        Prop = incl[i-1, :]
        Prop[ind] = !Prop[ind] # and permute sampled index

        # get ratio of marginal likelihoods times priors (in logs)
        Post_Prop = marginal_likelihood(y, [x W[:, Prop]], g) + model_prior(Prop, k; a = 1, m = m)
        Post_Curr = marginal_likelihood(y, [x W[:, Curr]], g) + model_prior(Curr, k; a = 1, m = m)

        # MH step
        alpha = min(1, exp(Post_Prop - Post_Curr))
        if alpha > rand(Uniform())
            incl[i,:] = Prop
            draw = post_sample(y, x, W[:, Prop], g)
            α_store[i] = draw.α
            τ_store[i] = draw.τ
            β_store[i,Prop] = draw.β
            σ_store[i] = draw.σ
        else
            incl[i,:] = Curr
            draw = post_sample(y, x, W[:, Curr], g)
            α_store[i] = draw.α
            τ_store[i] = draw.τ
            β_store[i,Curr] = draw.β
            σ_store[i] = draw.σ
        end
    end

    return (α = α_store[(burn+1):end],
            τ = τ_store[(burn+1):end],
            β = β_store[(burn+1):end,:],
            σ = σ_store[(burn+1):end],
            L = incl[(burn+1):end,:])
end