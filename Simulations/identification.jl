
using Distributions, LinearAlgebra
using StatsPlots, Plots

"""
Inspect the joint posterior of τ and δ
"""

# Joint posterior of tau and delta
function post_τ_δ(τ, δ; y = y, x = x, z = z, Σ = Σ, κ2_τ = 100, κ2_δ = 100)
    cov_ratio = Σ[1,2]/Σ[2,2]
    ψ2 = Σ[1,1] - Σ[1,2]^2/Σ[2,2]

    mean_τ = x' * (y - cov_ratio * (x - δ * z)) / (x'x + ψ2/κ2_τ)
    cov_τ = ψ2 * inv(x'x + ψ2/κ2_τ)

    a = Σ[1,2]^2/(Σ[2,2] * ψ2) + 1
    A = a/Σ[2,2] * x'z + 1/κ2_δ + ψ2 / (x'x + ψ2/κ2_τ) * cov_ratio^2 * (x'z)^2
    B = x' * (y - cov_ratio * x)

    mean_δ = (a/Σ[2,2] * x'z - cov_ratio * y'x - cov_ratio * B * x'z) / A

    res = logpdf(Normal(mean_τ, sqrt(cov_τ)), τ) + logpdf(Normal(mean_δ, sqrt(1/A)), δ)
    return res
end


# Joint posterior of τ and δ
function post_τ_δ(τ, δ; y = y, x = x, z = z, Σ = Σ, κ2_τ = 100, κ2_δ = 100)
    cov_ratio = Σ[1,2]/Σ[2,2]
    ψ2 = Σ[1,1] - Σ[1,2]^2/Σ[2,2]

    mean_τ = x' * (y - cov_ratio * (x - δ * z)) / (x'x + ψ2/κ2_τ)
    cov_τ = ψ2 * inv(x'x + ψ2/κ2_τ)

    a = Σ[1,2]^2/(Σ[2,2] * ψ2) + 1
    A = a/Σ[2,2] * x'z + 1/κ2_δ + ψ2 / (x'x + ψ2/κ2_τ) * cov_ratio^2 * (x'z)^2
    B = x' * (y - cov_ratio * x)

    mean_δ = (a/Σ[2,2] * x'z - cov_ratio * y'x - cov_ratio * B * x'z) / A

    res = logpdf(Normal(mean_τ, sqrt(cov_τ)), τ) + logpdf(Normal(mean_δ, sqrt(1/A)), δ)
    return res
end

# Define a function to generate a 3D surface plot of the joint density
function plot_joint_density(τ_range, δ_range)
    # Create a grid of points
    τ_values = range(τ_range[1], τ_range[2], length=100)
    δ_values = range(δ_range[1], δ_range[2], length=100)
    
    # Initialize a matrix to hold the density values
    Z = [post_τ_δ(τ, δ) for τ in τ_values, δ in δ_values]  # Exponentiate the log-density

    # Create a 3D surface plot
    surface(
        τ_values, δ_values, Z, alpha = 0.8,
        xlabel="τ", ylabel="δ", zlabel="Log-density", title="Joint posterior of τ and δ"
        )
end

n = 10000; c = 0.5
τ = 1/2; δ = 1; β = 0
z = rand(Normal(0, 1), n)
u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
x = z * δ  + u[:,2]
y = τ * x + β * z + u[:,1]
Σ = [1 c; c 1]

τ_range = (-1, 1)  # Range of τ values
δ_range = (-1, 1)  # Range of δ values
plot_joint_density(τ_range, δ_range)


"""
    Conditional distribution of τ given δ
"""
function post_τ(δ; y = y, x = x, z = z, Σ = Σ)
    cov_ratio = Σ[1,2]/Σ[2,2]
    ψ2 = Σ[1,1] - Σ[1,2]^2/Σ[2,2]

    mean_τ = inv(x'x) * x' * (y - cov_ratio * (x - δ * z))
    cov_τ = ψ2 * inv(x'x)

    res = Normal(mean_τ, sqrt(cov_τ))
    return res
end

n = 10000; c = 0.9
τ = 1/2; δ = 0.25
z = rand(Normal(0, 1), n)
u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
x = z * δ + u[:,2]
y = τ * x + u[:,1]
Σ = [1 c; c 1]

cov(x, (y-τ*x))
cov((x - δ*z), (y-τ*x))

inv(x'x) * x'y
inv(x'x) * x'z


plot(post_τ(0.5), label = "δ = 0.5")
plot!(post_τ(0.25), label = "δ = 0.25")
plot!(post_τ(0), label = "δ = 0")


"""
Inspect the posterior of the covariance matrix Σ given τ and δ
"""
n = 1000; p = 1; c = 0.5
τ = 1/2
δ = 1
z = rand(Normal(0, 1), n)

u = rand(MvNormal([0, 0], [1 c; c 1]), n)'
x = z * δ + u[:,2]
y = τ * x + z+ u[:,1]

Σ = [1 c; c 1]

function approx_post_Σ(τ, δ; y = y, x = x, z = z, m = 10000)
    ϵ = y - τ * x
    η = x - δ * z

    Q = [ϵ η]' * [ϵ η]
    ν = 3; n = length(y)
    Σ = rand(InverseWishart(ν + n, I + Q), m)

    res = map(x -> x[1,2]/x[2,2], Σ)
    return res
end

density(approx_post_Σ(1/2, 1))
density!(approx_post_Σ(1/2, 0))
vline!([c], label = "true value")