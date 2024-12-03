
using Distributions, LinearAlgebra, GLMNet, JuMP, RCall

"""
    This function implements a TSLS estimator to compare our approach to.
"""
function tsls(y, x, Z, W, y_h, x_h, W_h; level = 0.05)
    n = length(y)

    U = [ones(n) x W]
    V = [ones(n) Z W]  
    P_V = V * inv(V'V) * V'
    
    β_hat = inv(U' * P_V * U) * U' * P_V * y
    τ_hat = β_hat[2]

    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = sqrt(σ2_hat * inv(U' * P_V * U)[2, 2])
    ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat

    # compute lps on holdout dataset
    U_h = [ones(length(y_h)) x_h W_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (τ = τ_hat, CI = ci, lps = lps)
end

function tsls(y, x, Z, y_h, x_h, Z_h; level = 0.05)
    n = length(y)

    U = [ones(n) x]
    V = [ones(n) Z]  
    P_V = V * inv(V'V) * V'
    
    β_hat = inv(U' * P_V * U) * U' * P_V * y
    τ_hat = β_hat[2]

    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = sqrt(σ2_hat * inv(U' * P_V * U)[2, 2])
    ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat

    # compute lps on holdout dataset
    U_h = [ones(length(y_h)) x_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (τ = τ_hat, CI = ci, lps = lps)
end

"""
    OLS estimator.
"""
function ols(y, x, W; level = 0.05)
    n = length(y)
    U = [ones(n) x W]
    k = size(U, 2)

    ols = inv(U'U) * U'y
    τ_hat = ols[2]
    σ = sqrt( (y - U*ols)' * (y - U*ols) / (n-k) )
    cov = σ^2 * inv(U'U)
    sd_τ_hat = sqrt(cov[2,2])
    ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat

    return (τ = τ_hat, CI = ci)
end


"""
    This function implements the post-lasso estimator using a first-stage lasso to select the instruments.
"""
function post_lasso(y, x, Z, W, y_h, x_h, W_h; level = 0.05, sim = true)
    n = length(y)

    # First stage: Lasso regression of x on Z and W
    V = [Z W]  # Instruments and control variables
    fit_lasso = glmnetcv(V, x; alpha = 1.0, nfolds = 5)  # alpha=1 is the pure Lasso
    
    # Get the coefficients from the Lasso fit
    idx = argmin(fit_lasso.meanloss)
    coef_lasso = fit_lasso.path.betas[:, idx]

    # Select instruments with non-zero coefficients (ignoring intercept)
    selected_instruments = findall(coef_lasso[1:size(Z, 2)] .!= 0)

    # If no instruments are selected, decrease lambda until at least one is selected
    while isempty(selected_instruments) && idx < length(fit_lasso.path.lambda)
        idx += 1
        coef_lasso = fit_lasso.path.betas[:, idx]
        selected_instruments = findall(coef_lasso[1:size(Z, 2)] .!= 0)
    end

    # Subset the instruments based on Lasso selection
    Z_selected = Z[:, selected_instruments]
    
    # Combine selected instruments with the controls for the second stage
    V = [ones(n) Z_selected W]   
    P_V = V * inv(V'V) * V'
    U = [ones(n) x W]

    # Second stage: TSLS regression of y on x (using selected instruments and W)
    β_hat = inv(U' * P_V * U) * U' * P_V * y
    τ_hat = β_hat[2]

    # Compute residuals and standard errors
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = sqrt(σ2_hat * inv(U' * P_V * U)[2, 2])

    # Confidence interval for τ_hat
    ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat

    # Compute lps on holdout dataset
    U_h = [ones(length(y_h)) x_h W_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    if sim
        return (τ = τ_hat, CI = ci, lps = lps)
    else 
        return (τ = τ_hat, CI = ci, lps = lps, selected_instruments = selected_instruments)
    end
end


"""
    This implements a TSLS estimator with a Jackknife approximation in the first stage .
"""
function jive(y, x, Z, W, y_h, x_h, W_h; level = 0.05)
    n = length(y)

    # First stage: Initialize vector for fitted values (Jackknife approach)
    x_hat_jackknife = zeros(n)

    for i in 1:n
        # Remove the ith observation for Jackknife step
        Z_i = Z[setdiff(1:n, i), :]
        W_i = W[setdiff(1:n, i), :]
        x_i = x[setdiff(1:n, i)]
        
        # Instrumental regression: x ~ Z + W (without ith observation)
        V_i = [ones(n-1) Z_i W_i]
        beta_hat_i = inv(V_i' * V_i) * V_i' * x_i
        
        # Predict x_hat for the ith observation
        V_i_ith = [1; Z[i, :]; W[i, :]]
        x_hat_jackknife[i] = V_i_ith' * beta_hat_i
    end

    # Second stage: Run regression of y on x_hat_jackknife and W
    U_jive = [ones(n) x_hat_jackknife W]
    β_hat = inv(U_jive' * U_jive) * U_jive' * y
    τ_hat = β_hat[2]

    # Compute residuals and standard errors
    U = [ones(n) x W]
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = sqrt(σ2_hat * inv(U' * U)[2, 2])

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

    # First stage: Initialize vector for fitted values (Jackknife approach)
    x_hat_jackknife = zeros(n)

    for i in 1:n
        # Remove the ith observation for Jackknife step
        Z_i = Z[setdiff(1:n, i), :]
        W_i = W[setdiff(1:n, i), :]
        x_i = x[setdiff(1:n, i)]
        
        # Instrumental regression: x ~ Z + W (without ith observation)
        V_i = [ones(n-1) Z_i W_i]
        beta_hat_i = inv(V_i' * V_i + λ * I) * V_i' * x_i
        
        # Predict x_hat for the ith observation
        V_i_ith = [1; Z[i, :]; W[i, :]]
        x_hat_jackknife[i] = V_i_ith' * beta_hat_i
    end

    # Second stage: Run regression of y on x_hat_jackknife and W
    U_jive = [ones(n) x_hat_jackknife W]
    β_hat = inv(U_jive' * U_jive) * U_jive' * y
    τ_hat = β_hat[2]

    # Compute residuals and standard errors
    U = [ones(n) x W]
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = sqrt(σ2_hat * inv(U' * U)[2, 2])

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
    
    res = MA2SLS_raw(y, W, x, Z)
    
    β_hat = res[1]
    τ_hat = β_hat[end]

    # Compute residuals and standard errors
    U = [ones(n) W x]
    residuals = y - U * β_hat
    σ2_hat = sum(residuals.^2) / (n - size(U, 2))
    sd_τ_hat = res[2][end]

    # Confidence interval for τ_hat
    ci = τ_hat .+ [-1, 1] * quantile(Normal(0, 1), 1 - level/2) * sd_τ_hat

    # Compute lps on holdout dataset
    U_h = [ones(length(y_h)) W_h x_h]
    lps = -logpdf(MvNormal(U_h * β_hat, σ2_hat * I), y_h) / length(y_h)

    return (τ = τ_hat, CI = ci, lps = lps)
end


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
function ivbma_kl(y, X, Z, W, y_h, X_h, Z_h, W_h)
    n, l = (size(X, 1), size(X, 2))

    @rput y X Z W
    R"""
    source("ivbma.R")
    res = ivbma(y, X, Z, cbind(rep(1, nrow(W)), W), print.every = 1e5)
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

    return (
        τ = mean(τ, dims = 1)[1, :],
        CI = [quantile(τ[:, i], [0.025, 0.975]) for i in axes(τ, 2)],
        lps = lps
    )
end

# alternative method for invalid instruments
ivbma_kl(y, X, Z, y_h, X_h, Z_h) = ivbma_kl(y, X, Matrix{Float64}(undef, length(y), 0), Z, y_h, X_h, Matrix{Float64}(undef, length(y_h), 0), Z_h)

