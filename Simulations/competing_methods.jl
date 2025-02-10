
using Distributions, LinearAlgebra, GLMNet, JuMP, RCall

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
    sd_τ_hat = sqrt(σ2_hat * inv(U_jive' * U_jive)[2, 2])

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
    sd_τ_hat = sqrt(σ2_hat * inv(U_jive' * U_jive)[2, 2])

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
function ivbma_kl(y, X, Z, W, y_h, X_h, Z_h, W_h; target_M = [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0])
    n, l = (size(X, 1), size(X, 2))

    @rput y X Z W
    R"""
    source("ivbma.R")
    max_attempts <- 5
    attempt <- 1
    # It sometimes happens that the ivbma code fails because of a singular matrix (this is usually within the Bartlett decomposition). We just retry it a few times (this rarely happens so should not affect the results too much)
    while(attempt <= max_attempts) {
        tryCatch({
            res <- ivbma(y, X, Z, cbind(rep(1, nrow(W)), W), print.every = 1e5)
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
    α = res[:rho][:, l+1]
    τ = res[:rho][:, 1:l]
    β = res[:rho][:, (l+2):end]
    Λ = res[:lambda]
    Σ = res[:Sigma]

    # Compute LPS on holdout data
    n_h = length(y_h)
    scores = Matrix{Float64}(undef, n_h, length(α))
    for i in eachindex(α)
        H = X_h - [ones(n_h) Z_h W_h] * Λ[:, :, i]
        mean_y = α[i] * ones(n_h) + X_h[:, :] * τ[i, :] + W_h * β[i, :] + H * inv(Σ[2:end, 2:end, i]) * Σ[2:end, 1, i]
        σ_y_x = Σ[1, 1, i] - Σ[2:end, 1, i]' * inv(Σ[2:end, 2:end, i]) * Σ[2:end, 1, i]

        scores[:, i] = [pdf(Normal(mean_y[j], sqrt(σ_y_x)), y_h[j]) for j in eachindex(y_h)]
    end
    scores_avg = mean(scores; dims = 2)
    lps = -mean(log.(scores_avg))

    # compute posterior probability of true treatment model (this only matters for the simulation with multiple endogenous variables; we do not use this in the other scenarios)
    M = res[:M][2:end, :, :]
    posterior_probability_M = [mean(mapslices(slice -> slice[:, 1] == target_M, M, dims=[1,2])), mean(mapslices(slice -> slice[:, 2] == target_M, M, dims=[1,2]))]

    # compute mean model size (if l>1 we average the model sizes for the different endogenous variables)
    M_size_bar = mean(sum(M, dims = 1), dims = 3)[1, :, 1]

    return (
        τ = l == 1 ? mean(τ) : mean(τ, dims = 1)[1, :],
        CI = l == 1 ? quantile(τ, [0.025, 0.975]) : [quantile(τ[:, i], [0.025, 0.975]) for i in axes(τ, 2)],
        lps = lps,
        posterior_probability_M = posterior_probability_M,
        M_bar = res[:M_bar][2:end, :]',
        M_size_bar = M_size_bar,
        L = res[:L_bar][2:end]
    )
end

# alternative method for invalid instruments
ivbma_kl(y, X, Z, y_h, X_h, Z_h) = ivbma_kl(y, X, Matrix{Float64}(undef, length(y), 0), Z, y_h, X_h, Matrix{Float64}(undef, length(y_h), 0), Z_h)

