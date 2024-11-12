
using StatsPlots, Integrals, SpecialFunctions

# Inspect marginal prior of α for different values of ν and g
plot(Normal(0, 10), xlim = (-20, 20), label = "N(0, 10^2)", xlabel = "α")
plot!(Normal(0, 5), xlim = (-20, 20), label = "N(0, 5^2)")

function plot_prior(g, ν, n)
        α = Vector(undef, 1000)
        σ_y_x = Vector(undef, 1000)
        for i in eachindex(α)
                Σ = rand(InverseWishart(ν, [1 0 0; 0 1 0; 0 0 1]))
                Σ_xx = Σ[2:end, 2:end]
                Σ_yx = Σ[2:end, 1]
                σ_y_x[i] = Σ[1,1] - Σ_yx' * inv(Σ_xx) * Σ_yx

                α[i] = rand(Normal(0, σ_y_x[i] * g / n))
        end

        density!(α, label = "G_prior(g/n = $(g/n), ν = $ν)")
end

plot_prior(100, 4, 100)
plot_prior(500, 4, 100)
plot_prior(100, 3, 100)
plot_prior(500, 3, 100)


# hyperprior on ν
jp_ν(ν, p) = ((ν+1)/(ν+3))^(p/2) * (ν/(ν+3))^(1/2) * (SpecialFunctions.trigamma(ν/2) -  SpecialFunctions.trigamma((ν+1)/2) - 2*(ν+3)/(ν*(ν+1)^2))^(1/2)

function jp_ν_normalised(ν, p)
        domain = (1, Inf)
        prob = IntegralProblem(jp_ν, domain, p)
        sol = solve(prob, QuadGKJL())
        
        return jp_ν(ν, p) / sol.u
end

x = 1:0.1:50
p = plot(x, jp_ν_normalised.(x, 10), label = "p = 10", xlabel = "ν", ylabel = "p(ν)")
plot!(x, jp_ν_normalised.(x, 25), label = "p = 25")
plot!(x, jp_ν_normalised.(x, 50), label = "p = 50")
savefig(p, "Priors/Hyperprior_nu.pdf")
