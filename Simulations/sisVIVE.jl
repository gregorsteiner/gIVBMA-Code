
using Distributions, LinearAlgebra, RCall

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
