##### Bayes factor simulation #####

using Distributions, LinearAlgebra, Random, ProgressBars
using CairoMakie, LaTeXStrings
using Pkg; Pkg.activate("../../gIVBMA")
using gIVBMA


# data generating function
function gen_data(n, c, τ)
    ι = ones(n)
    Z_10 = rand(MvNormal(zeros(10), I), n)'
    Z_15 = Z_10[:, 1:5] * [0.3, 0.5, 0.7, 0.9, 1.1] * ones(5)' + rand(Normal(0, 1), n, 5)
    Z = [Z_10 Z_15]

    Σ = c .^ abs.((1:3) .- (1:3)') ./ 2
    u = rand(MvNormal([0, 0, 0], Σ), n)'

    Q = ι * [4, -1]' + Z[:, 1] * [2, -2]' + Z[:, 5] * [-1, 1]' + Z[:, 7] * [1.5, 1]' + Z[:, 11] *[1, 1]' + Z[:, 13] * [1/2, -1/2]' + u[:, 2:3] 
    X = Q

    y = ι + X * τ + u[:, 1]

    return (y = y, X = X, Z = Z)
end

# function that runs the simulation
function sim_func(m, n)
    BFs = Matrix{Float64}(undef, 3, m)

    for i in ProgressBar(1:m)
        y, X, Z = gen_data(n, 1/2, [1/2, -1/2])
        res_full = givbma(y, X, Z; g_prior = "hyper-g/n")
        res_red_1 = givbma(y, X, Z[:, 1:13]; g_prior = "hyper-g/n") # drop instruments 14 and 15 which are irrelevant
        res_red_2 = givbma(y, X, Z[:, 2:end]; g_prior = "hyper-g/n") # drop the first instrument (which is relevant)
        res_red_3 = givbma(y, X[:, 2], [X[:, 1] Z]; g_prior = "hyper-g/n") # drop the first endogenous variable and use it as exogenous covariate
        
        BFs[1, i] = res_red_1.ML_outcome - res_full.ML_outcome + res_red_1.ML_treatment - res_full.ML_treatment 
        BFs[2, i] = res_red_2.ML_outcome - res_full.ML_outcome + res_red_2.ML_treatment - res_full.ML_treatment
        BFs[3, i] = res_red_3.ML_outcome - res_full.ML_outcome
    end
    return BFs
end

Random.seed!(42)
m = 100
res50 = sim_func(m, 50)
res500 = sim_func(m, 500)

fig = Figure()

ax1 = Axis(fig[1, 1], title = "a)", ylabel = L"\log(BF_{01})", xticks = (1:2, [L"n = 50", L"n = 500"]))
boxplot!(ax1, repeat([1, 2], inner = m), [res50[1, :] res500[1, :]])

ax2 = Axis(fig[1, 2], title = "b)", ylabel = "", xticks = (1:2, [L"n = 50", L"n = 500"]))
boxplot!(ax2, repeat([1, 2], inner = m), [res50[2, :] res500[2, :]])

ax3 = Axis(fig[1, 3], title = "c)", ylabel = "", xticks = (1:2, [L"n = 50", L"n = 500"]))
boxplot!(ax3, repeat([1, 2], inner = m), [res50[3, :] res500[3, :]])

save("BF_Simulation.pdf", fig)


