
using Distributions, LinearAlgebra, GLMNet, JuMP, RCall
using Turing, DynamicPPL, AdvancedVI
using Turing: Variational
using Bijectors: bijector
using Logging
Logging.disable_logging(Logging.Info)


include("aux_functions.jl")

"""
    This function implements a TSLS estimator to compare our approach to.
"""
function tsls(y, x, Z, W, y_h, x_h, W_h; level = 0.05)
    n = length(y)
    l = size(x, 2)

    U = [ones(n) x W]
    V = [ones(n) Z W]  
    P_V = V * inv(V'V) * V'
    
    β_hat = inv(U' * P_V * U) * U' * P_V * y
    τ_hat = β_hat[2:(l+1)]

    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    cov = σ2_hat * inv(U' * P_V * U)
    cov_τ = cov[2:(l+1), 2:(l+1)]
    ci = [τ_hat[j] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov_τ[j, j]) for j in eachindex(τ_hat)]

    # compute lps on holdout dataset
    U_h = [ones(length(y_h)) x_h W_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (
        τ = l == 1 ? τ_hat[1] : τ_hat,
        CI = l == 1 ? ci[1] : ci,
        lps = lps
    )
end

tsls(y, x, Z, y_h, x_h, Z_h; level = 0.05) = tsls(y, x, Z, Matrix{Float64}(undef, length(y), 0), y_h, x_h, Matrix{Float64}(undef, length(y_h), 0); level = level)

"""
    OLS estimator.
"""
function ols(y, X, W, y_h, X_h, W_h; level = 0.05)
    n = length(y)
    l = size(X, 2)
    U = [ones(n) X W]

    β_hat = inv(U'U) * U'y
    τ_hat = β_hat[2:(l+1)]

    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    cov = σ2_hat * inv(U'U)
    cov_τ = cov[2:(l+1), 2:(l+1)]
    
    ci = [τ_hat[j] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sqrt(cov_τ[j, j]) for j in eachindex(τ_hat)]

    U_h = [ones(length(y_h)) X_h W_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (
        τ = l == 1 ? τ_hat[1] : τ_hat,
        CI = l == 1 ? ci[1] : ci,
        lps = lps
    )
end


"""
    This function implements the post-lasso estimator using a first-stage lasso to select the instruments.
"""
function post_lasso(y, x, Z, W, y_h, x_h, W_h; level = 0.05)
    @rput y x Z W
    R"res = suppressMessages(hdm::rlassoIV(W, x, y, Z))" # The messages are suppressed for convenience, we still see that no instruments were selected when checking the standard error below
    @rget res

    # if no instruments are selected, we just return missing values
    if ismissing(res[:se])
        return (τ = missing, CI = [missing, missing], lps = missing, no_instruments = true)
    else
        τ_hat = res[:coefficients]
        sd_τ_hat = res[:se]
        ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat
        return (τ = τ_hat, CI = ci, lps = missing, no_instruments = false)
    end
end


"""
    This implements a TSLS estimator with a Jackknife approximation in the first stage .
"""
function jive(y, x, Z, W, y_h, x_h, W_h; level = 0.05)
    n = length(y)

    U = [ones(n) x W]
    U_jive = zeros(n, size(U, 2))
    V = [ones(n) Z W]

    # First stage: Obtain jackknife-predicted values
    for i in 1:n
        V_i = V[setdiff(1:n, i), :]
        U_i = U[setdiff(1:n, i), :]
        beta_hat_i = inv(V_i' * V_i) * V_i' * U_i
        U_jive[i, :] = V[i, :]' * beta_hat_i
    end

    # Second stage: Run regression of y on jackknife U
    β_hat = inv(U_jive' * U) * U_jive' * y
    τ_hat = β_hat[2]

    # Compute residuals and standard errors
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = sqrt(σ2_hat * inv(U' * U_jive * inv(U_jive' * U_jive) * U_jive' * U)[2, 2])

    # Confidence interval for τ_hat
    ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat

    # Compute lps on holdout dataset
    U_h = [ones(length(y_h)) x_h W_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (τ = τ_hat, CI = ci, lps = lps)
end


"""
    This implements a TSLS estimator with a Jackknife approximation and Ridge regularisation in the first stage .
"""
function rjive(y, x, Z, W, y_h, x_h, W_h; level = 0.05)
    n = length(y)

    p = size(Z, 2)
    M_W = I - W * inv(W'W) * W'
    λ = var(M_W * x) * p # recommended choice in Hansen & Kozbur (2014)

    U = [ones(n) x W]
    U_jive = zeros(n, size(U, 2))
    V = [ones(n) Z W]

    # First stage: Obtain jackknife-predicted values
    for i in 1:n
        V_i = V[setdiff(1:n, i), :]
        U_i = U[setdiff(1:n, i), :]
        beta_hat_i = inv(V_i' * V_i + λ * I) * V_i' * U_i
        U_jive[i, :] = V[i, :]' * beta_hat_i
    end

    # Second stage: Run regression of y on x_hat_jackknife and W
    β_hat = inv(U_jive' * U) * U_jive' * y
    τ_hat = β_hat[2]

    # Compute residuals and standard errors
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = sqrt(σ2_hat * inv(U' * U_jive * inv(U_jive' * U_jive) * U_jive' * U)[2, 2])

    # Confidence interval for τ_hat
    ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat

    # Compute lps on holdout dataset
    U_h = [ones(length(y_h)) x_h W_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (τ = τ_hat, CI = ci, lps = lps)
end

"""
    This function implements the MA2SLS estimator by Kuersteiner & Okui (2010), Econometrica
"""

include("MA2SLS.jl")

function matsls(y, x, Z, W, y_h, x_h, W_h; level = 0.05)
    n = length(y)
    l = size(x, 2)
    
    res = MA2SLS_raw(y, W, x, Z)

    β_hat = res[1]
    τ_hat = β_hat[(end-(l-1)):end] # the last l elements of the coefficient vector are estimates of τ

    # Compute residuals and standard errors
    U = [ones(n) W x]
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = res[2][(end-(l-1)):end]

    # compute CI
    ci = [τ_hat[j] .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat[j] for j in eachindex(τ_hat)]

    # Compute lps on holdout dataset
    U_h = [ones(length(y_h)) W_h x_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (
        τ = l == 1 ? τ_hat[1] : τ_hat,
        CI = l == 1 ? ci[1] : ci,
        lps = lps
    )
end

matsls(y, x, Z, y_h, x_h, Z_h; level = 0.05) = matsls(y, x, Z, Matrix{Float64}(undef, length(y), 0), y_h, x_h, Matrix{Float64}(undef, length(y_h), 0); level = level)


"""
    This function implements the sisVIVE estimator by Kang et al (2016), JASA.
"""
sisVIVE = function(y, x, Z, y_h, x_h, Z_h; level = 0.05)
    n = length(y)

    @rput y x Z
    R"res = sisVIVE::cv.sisVIVE(y, x, Z, K = 5)"
    @rget res
    
    β_hat = [0; res[:beta]; res[:alpha]]
    τ_hat = β_hat[2]

    U = [ones(n) x Z]
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))

    # sisVIVE does not provide standard errors so we set the coverage to zero
    ci = [τ_hat, τ_hat]

    # Compute LPD on holdout dataset
    U_h = [ones(length(y_h)) x_h Z_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (τ = τ_hat, CI = ci, lps = lps)
end


"""
    Implement the IVBMA procedure of Karl & Lenkoski based on their R package.
"""
function ivbma_kl(y, X, Z, W, y_h, X_h, Z_h, W_h; s = 2000, b = 1000, target_M = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], extract_instruments = true)
    n, l = (size(X, 1), size(X, 2))

    # centre data (so we do not need intercepts)
    # We centre the holdout data with the training mean (for the LPS calculation)
    y_c = y .- mean(y)
    X_c = X .- mean(X, dims = 1)
    y_h = y_h .- mean(y)
    X_h = X_h .- mean(X, dims = 1)

    @rput y_c X_c Z W s b
    R"""
    source("ivbma.R")
    max_attempts <- 5
    attempt <- 1
    # It sometimes happens that the ivbma code fails because of a singular matrix (this is usually within the Bartlett decomposition).
    # We just retry it a few times (this rarely happens so should not affect the results too much)
    while(attempt <= max_attempts) {
        tryCatch({
            res <- ivbma(y_c, X_c, Z, W, s = s, b = b, odens = s-b, print.every = 1e5)
            break  # If successful, exit the while loop
        }, error = function(e) {
            if(attempt == max_attempts) {
                stop("Failed after ", max_attempts, " attempts. Last error: ", e$message)
            }
            attempt <- attempt + 1
        })
    }
    """
    @rget res
    # Store posterior sample
    τ = res[:rho][:, 1:l]
    β = res[:rho][:, (l+1):end]
    Λ = res[:lambda]
    Σ = res[:Sigma]

    # Compute LPS on holdout data
    n_h, n_post = length(y_h), size(β, 1)
    scores = Matrix{Float64}(undef, n_h, n_post)
    for i in 1:n_post
        H = X_h - [Z_h W_h] * Λ[:, :, i]
        mean_y = X_h[:, :] * τ[i, :] + W_h * β[i, :] + H * inv(Σ[2:end, 2:end, i]) * Σ[2:end, 1, i]
        σ_y_x = Σ[1, 1, i] - Σ[2:end, 1, i]' * inv(Σ[2:end, 2:end, i]) * Σ[2:end, 1, i]

        scores[:, i] = [pdf(Normal(mean_y[j], sqrt(σ_y_x)), y_h[j]) for j in eachindex(y_h)]
    end
    scores_avg = mean(scores; dims = 2)[:, 1]
    scores_avg = ifelse.(scores_avg .== 0, 1e-300, scores_avg) # if any of the scores is numerically zero, we set it to 1e-300 such that its log is not -Inf
    lps = -mean(log.(scores_avg))

    # compute posterior probability of true treatment model (this only matters for the simulation with multiple endogenous variables; we do not use this in the other scenarios)
    M = res[:M][:, :, :]
    if l > 1 # the line below only makes sens if l is at least 2
        posterior_probability_M = [mean(mapslices(slice -> slice[:, 1] == target_M, M, dims=[1,2])), mean(mapslices(slice -> slice[:, 2] == target_M, M, dims=[1,2]))]
    else # else this doesn't matter and we'll just assign (0, 0), but won't use this
        posterior_probability_M = [0, 0]
    end

    # compute mean model size (if l>1 we average the model sizes for the different endogenous variables)
    M_size_bar = mean(sum(M, dims = 1), dims = 3)[1, :, 1]

    # Extract the number of instruments (if needed)
    # In L, we drop the first l variables (treatments)
    # We select M for the first variable as we only use this for l = 1
    # This is not needed in all scenarios => wrap it in this if-statement
    if extract_instruments
        N_Z = extract_instruments(res[:L]'[(l+1):(end), :], res[:M][:, 1, :])
    else
        N_Z = missing
    end

    return (
        τ = l == 1 ? mean(τ) : mean(τ, dims = 1)[1, :],
        CI = l == 1 ? quantile(τ, [0.025, 0.975]) : [quantile(τ[:, i], [0.025, 0.975]) for i in axes(τ, 2)],
        lps = lps,
        L = res[:L],
        M = res[:M],
        posterior_probability_M = posterior_probability_M,
        M_bar = res[:M_bar]',
        M_size_bar = M_size_bar,
        L_bar = res[:L_bar],
        τ_full = res[:rho][:, 1:l],
        Σ = Σ,
        N_Z = N_Z
    )
end

# alternative method for invalid instruments
ivbma_kl(y, X, Z, y_h, X_h, Z_h; s = 2000, b = 1000) = ivbma_kl(y, X, Matrix{Float64}(undef, length(y), 0), Z, y_h, X_h, Matrix{Float64}(undef, length(y_h), 0), Z_h; s = s, b = b)


"""
    A global-local shrinkage approach to achieve shrinkage in both equations. Currently only works for a single endogenous variable (l=1).
"""
@model function HorseshoeBayesianIV(y, x, Z, W)
    p1, p2 = size(Z, 2), size(W, 2)

    # Covariance prior
    ν_transf ~ Exponential(1)
    ν = ν_transf + 2
    Σ_xx ~ InverseGamma((ν-1)/2, 1/2)
    σ_y_x ~ InverseGamma(ν/2, 1/2)
    a ~ Normal(0, sqrt(σ_y_x))
    #A = [1.0 a; 0.0 1.0]
    #Σ = A * [σ_y_x 0.0; 0.0 Σ_xx] * A'


    # Intercepts and treatment effect priors
    α ~ Normal(0, 10)
    τ ~ Normal(0, 10)
    γ ~ Normal(0, 10)

    # Horseshoe
    halfcauchy  = truncated(Cauchy(0, 1); lower=0)
    τ_or ~ halfcauchy
    λ_or ~ filldist(halfcauchy, p2)
    τ_tr ~ halfcauchy
    λ_tr ~ filldist(halfcauchy, p1+p2)

    #β_inner ~ MvNormal(zeros(p), I)
    #β = β_inner .* λ_or * τ_or
    #Turing.@addlogprob! -sum(log.(λ_or * τ_or))
    β ~ MvNormal(zeros(p2), I * λ_or.^2 * τ_or^2)

    #δ_inner ~ MvNormal(zeros(p), I)
    #δ = δ_inner .* λ_tr * τ_tr
    #Turing.@addlogprob! -sum(log.(λ_tr * τ_tr))
    δ ~ MvNormal(zeros(p1+p2), I * λ_tr.^2 * τ_tr^2)

    # likelihood
    y ~ MvNormal(α .+ x * τ + W * β + (x .- γ - [Z W] * δ) * a, σ_y_x * I)
    x ~ MvNormal(γ .+ [Z W] * δ, Σ_xx * I)
end


function hsiv(y, x, Z, W, y_h, x_h, Z_h, W_h; samples = 1000)
    # fit model
    model = HorseshoeBayesianIV(y, x, Z, W)
    
    q0 = q_meanfield_gaussian(model)
    q_avg, info, state = vi(model, q0, 500; show_progress=false)

    # sample from the variational approx to get summary statistics
    z = rand(q_avg, samples)

    # get parameter indices
    _, sym2range = bijector(model, Val(true));
    idx = map(x -> x[1], sym2range)

    # LPS calculation
    n_h = length(y_h)
    scores = Matrix{Float64}(undef, n_h, samples)
    for i in 1:samples
        σ_y_x, a = z[idx.σ_y_x[1], i], z[idx.a[1], i]
        α, τ, β = z[idx.α[1], i], z[idx.τ[1], i], z[idx.β, i]
        γ, δ = z[idx.γ[1], i], z[idx.δ, i]
        H = x_h .- γ - [Z_h W_h] * δ
        mean_q = α .+ x_h * τ + W_h * β + H * a
        scores[:, i] = [pdf(Normal(mean_q[j], sqrt(σ_y_x)), y_h[j]) for j in eachindex(y_h)]
    end
    scores_avg = mean(scores; dims = 2)
    lps = -mean(log.(scores_avg))

    return (
        τ = mean(z[idx.τ[1], :]),
        CI = quantile(z[idx.τ[1], :], [0.025, 0.975]),
        lps = lps
    )
end

## alternative method for free instrument selection
hsiv(y, x, Z, y_h, x_h, Z_h; samples = 1000) = hsiv(y, x, Matrix{Float64}(undef, length(y), 0), Z, y_h, x_h, Matrix{Float64}(undef, length(y_h), 0), Z_h; samples = samples)