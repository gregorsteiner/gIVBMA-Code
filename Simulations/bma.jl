using SpecialFunctions, Distributions, Statistics

"""
    This file implements a 'naive' BMA approach, i.e. only on the outcome model ignoring the treatment model.
    This corresponds to ignoring the endogeneity.
"""

function post_sample(y, X, W, g)
    n, l = size(X)
    
    y_bar = Statistics.mean(y)
    U = [X W]
    δ = g / (g+1)
    U_i = [ones(n) U]

    Q = I - U_i * inv(U_i'U_i) * U_i'

    s = δ * y' * Q * y + (1-δ) * (y .- y_bar)' * (y .- y_bar)
    σ = sqrt(rand(InverseGamma((n-1)/2, s/2))) 
    α = rand(Normal(y_bar, σ/sqrt(n))) 
    β_t = rand(MvNormal(δ * inv(U'U)*U'y, Symmetric(σ^2 * δ * inv(U'U))))
    τ = β_t[1:l]
    β = β_t[(l+1):end]
    
    return (α = α, τ = τ, β = β, σ = σ)
end

function marginal_likelihood(y, W, g)
    n = size(W, 1)
    k_j = size(W, 2)
    
    X_i = [ones(n) W]
    Q = I - X_i*inv(X_i'X_i)*X_i'
    
    invDetCoef = y'Q * y / dot((y .- mean(y)), (y .- mean(y)))
    
    res = ((n-1-k_j)/2) * log((1+g)) + (-(n-1)/2) * log(1 + g*(invDetCoef))
    return res
end

function model_prior(x, k; a = 1, m = floor(k/2))
    b = (k - m) / m 
    kj = sum(x)
    
    Γ(x) = SpecialFunctions.gamma(x)
    res = log(Γ(a+b)) - log(Γ(a) * Γ(b)) + log(Γ(a+kj) * Γ(b+k-kj)) - log(Γ(a+b+k))
    return res
end

hyper_g_n(g; a = 3, n = 100) = (a-2)/(2*n) * (1 + g/n)^(-a/2)

function adjust_variance(curr_variance, acc_prob, desired_acc_prob, iter)
    log_variance = log(curr_variance) + iter^(-0.6) * (acc_prob - desired_acc_prob)
    return exp(log_variance)
end

function bma(y::AbstractVector, X::AbstractMatrix, W::AbstractMatrix; iter::Integer = 2000, burn::Integer = 1000, g_prior = "BRIC", dist = "Gaussian")
    n, k = size(W)
    l = size(X, 2)
    
    X = X .- mean(X; dims = 1)
    W = W .- mean(W; dims = 1)

    g = max(n, k^2)
    g_random = (g_prior == "hyper-g/n")
    if g_random
        proposal_variance_g = 0.01
    end

    m = floor(k/2)

    L = sample([true, false], k, replace = true)
    α, τ, β, σ = (0, zeros(l), zeros(k)[L], 1)

    nsave = iter - burn
    α_store = zeros(nsave)
    τ_store = zeros(nsave, l)
    β_store = zeros(nsave, k)
    σ_store = zeros(nsave)
    L_store = Array{Bool}(undef, nsave, k)

    for i in 2:(iter)
        # draw proposal
        Prop = copy(L)
        ind = sample(1:k) # draw index to permute
        Prop[ind] = !Prop[ind] # and permute sampled index

        # get ratio of marginal likelihoods times priors (in logs)
        acc = min(1, exp(
            marginal_likelihood(y, [X W[:, Prop]], g) + model_prior(Prop, k; a = 1, m = m) -
            (marginal_likelihood(y, [X W[:, L]], g) + model_prior(L, k; a = 1, m = m))
        ))
        # MH step
        if rand() < acc
            L = Prop
        end

        # update g
        if g_random
            prop = rand(LogNormal(log(g), sqrt(proposal_variance_g)))
            acc = min(1, exp(
                marginal_likelihood(y, [X W[:, L]], prop) + log(hyper_g_n(prop; a = 3, n = n)) + log(prop) - 
                (marginal_likelihood(y, [X W[:, L]], g) + log(hyper_g_n(g; a = 3, n = n)) + log(g))
            ))
            if rand() < acc
                g = prop
            end
            proposal_variance_g = adjust_variance(proposal_variance_g, acc, 0.234, i)
        end

        # draw parameters
        α, τ, β, σ = post_sample(y, X, W[:, L], g)

        if i > burn
            α_store[i-burn] = α
            τ_store[i-burn, :] = τ
            β_store[i-burn, L] = β
            σ_store[i-burn] = σ
            L_store[i-burn, :] = L
        end


    end

    return (α = α_store,
            τ = τ_store,
            β = β_store,
            σ = σ_store,
            L = L_store)
end