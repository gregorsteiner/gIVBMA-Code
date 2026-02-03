# This file implements a simulation experiment
# where some of the mass is on a non-identified model

using Distributions
using LinearAlgebra
using Random
using gIVBMA

include("aux_functions.jl")

# simulate from the prior predictive for p = 1
function simulate_prior_predictive(n = 100; p = 1)
    
    # Generate instruments/covariates from standard normal
    Z = randn(n, p)
    
    # Draw model parameters from prior
    w_L = rand(Beta(1.0, 1.0))
    w_M = rand(Beta(1.0, 1.0))
    
    L = rand(Bernoulli(w_L))
    M = rand(Bernoulli(w_M))
    
    # fix g parameters to BRIC values
    g_L = max(n, (p + 2)^2)
    g_M = max(n, (p + 1)^2)
    
    # Draw structural covariance and residual
    nu = 2 + rand(Exponential(1.0))
    Sigma = rand(InverseWishart(nu, Matrix{Float64}(I, 2, 2)))
    σ_y_x = Sigma[1, 1] - Sigma[1, 2]^2 / Sigma[2, 2]
    u = rand(MvNormal(zeros(2), Sigma), n)

    # draw treatment parameters and treatment
    V = M ? [ones(n) Z] : ones(n)[:, :]
    Lambda = rand(MvNormal(zeros(size(V, 2)), g_M * Sigma[2, 2] * Symmetric(inv(V'V))))

    x = V * Lambda + u[2, :]

    # draw outcome prameters 
    U = L ? [ones(n) x Z] : [ones(n) x]
    rho = rand(MvNormal(zeros(size(U, 2)), g_L * σ_y_x * Symmetric(inv(U'U))))
    y = U * rho + u[1, :]
    
    return (
        y = y, x = x, Z = Z,
    )
end

# run experiment
function simulation_function(m, n)
    posterior_prob_non_identified = Vector{Float64}(undef, m)
    for i in eachindex(posterior_prob_non_identified)
        y, x, Z = simulate_prior_predictive(n)

        res = givbma(y, x, Z)
        N = extract_instruments(res.L, res.M)
        posterior_prob_non_identified[i] = mean(N .== 0)
    end
    return mean(posterior_prob_non_identified)
end

Random.seed!(42)
println(
    "Mean posterior probability of non-identification: ",
    round.(simulation_function(100, 100), digits = 3)
)


# compare to prior probability
function instrument_prior(n_z, p)
    total = 0.0
    Γ(x) = SpecialFunctions.gamma(x)
    for p_i in 0:p
        for k in 0:p_i
            if n_z <= (p - p_i)
                total += binomial(p_i, p_i - k) * binomial(p - p_i, n_z) * Γ(1 + p_i + n_z - k) * Γ(1 + p - p_i - n_z + k) / Γ(p + 2)
            end
        end
    end
    return total / (p+1)
end

println(
    "Prior probability of non-identification: ",
    instrument_prior(0, 1)
)

