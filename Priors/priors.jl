
using StatsPlots, Integrals, SpecialFunctions, Distributions


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


# What is the implied prior on \sigma_{12}?
using Turing

@model function ν_prior(p)
        ν ~ Uniform(1.5, 100)
        Turing.@addlogprob! log(jp_ν(ν, p)) 
end

function sample_σ12(p)
        chain = sample(ν_prior(p), NUTS(), 100000)
        ν = chain[:ν]
        res = map(x -> rand(InverseWishart(x, [1 0; 0 1]))[1, 2], ν)
        return res[-1/4 .< res .< 1/4] # for plotting purposes only return non-extreme values
end

pp = [2, 10, 25]
res = map(p -> sample_σ12(p), pp)

labels = permutedims("p = " .* string.(pp))
p_σ12 = density(res, label = labels, xlabel = "σ_12", ylabel = "Density")

res_ν3 = map(x -> x[1,2], rand(InverseWishart(3, [1 0; 0 1]), 100000))
density!(res_ν3[-1/2 .< res_ν3 .< 1/2], label = "Fixed ν = 3", xlim = (-0.25, 0.25))
res_ν10 = map(x -> x[1,2], rand(InverseWishart(10, [1 0; 0 1]), 100000))
density!(res_ν10[-1/2 .< res_ν10 .< 1/2], label = "Fixed ν = 10", xlim = (-0.25, 0.25))
savefig(p_σ12, "Implied_Prior_Sigma12.pdf")


# What about a shifted Exponential prior on nu?
function sample_σ12(λ, m = 100000)
        ν = rand(Exponential(λ), m) .+ 2
        Σ = map(x -> rand(InverseWishart(x, [1 0; 0 1])), ν)
        res = map(x -> x[1, 2] / x[2, 2], Σ)
        return res[-10 .< res .< 10] # for plotting purposes only return non-extreme values
end


p_exp = density(sample_σ12(0.1), xlim = (-5, 5),
                label = "Exponential(0.1)",
                xlabel = "σ₁₂ / σ₂₂",
                ylabel = "Density")
density!(sample_σ12(1), label = "Exponential(1)")
res_ν3 = map(x -> x[1,2]/x[2, 2], rand(InverseWishart(3, [1 0; 0 1]), 100000))
density!(res_ν3[-10 .< res_ν3 .< 10], label = "fixed ν = 3")
res_ν5 = map(x -> x[1,2]/x[2, 2], rand(InverseWishart(5, [1 0; 0 1]), 100000))
density!(res_ν5[-10 .< res_ν5 .< 10], label = "fixed ν = 5")
savefig(p_exp, "Implied_prior_with_exponential.pdf")
