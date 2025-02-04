using CairoMakie, LaTeXStrings
using Distributions, Random, LinearAlgebra, KernelDensity

##### This code creates a plot of the implied priors by the hyperprior on ν #####

# Define the Exponential priors
xx = 2:0.01:5
y1 = pdf.(Exponential(0.1), xx .- 2)
y2 = pdf.(Exponential(1), xx .- 2)
y3 = pdf.(Exponential(5), xx .- 2)

# Functions to sample from the implied priors
function sample_σ11(λ, m = 100000)
    ν = rand(Exponential(λ), m) .+ 2
    Σ = map(x -> rand(InverseWishart(x, [1 0; 0 1])), ν)
    res = map(x -> x[1, 1] - x[1, 2]^2 / x[2, 2], Σ)
    return res[-10 .< res .< 10] # drop very extreme values for the plot
end

function sample_σ12(λ, m = 100000)
    ν = rand(Exponential(λ), m) .+ 2
    Σ = map(x -> rand(InverseWishart(x, [1 0; 0 1])), ν)
    res = map(x -> x[1, 2] / x[2, 2], Σ)
    return res[-10 .< res .< 10]
end

# Generate data for the implied priors
data_exp12 = sample_σ12(0.1)
data_exp22 = sample_σ12(1)
data_exp32 = sample_σ12(5)
data_ν3 = map(x -> x[1, 2] / x[2, 2], rand(InverseWishart(3, [1 0; 0 1]), 100000))
data_ν3 = data_ν3[-10 .< data_ν3 .< 10]
data_ν5 = map(x -> x[1, 2] / x[2, 2], rand(InverseWishart(5, [1 0; 0 1]), 100000))
data_ν5 = data_ν5[-10 .< data_ν5 .< 10]

data_exp11 = sample_σ11(0.1)
data_exp21 = sample_σ11(1)
data_exp31 = sample_σ11(5)

# set up colour palette
cols = Makie.wong_colors()

# Create the figure and axis
fig = Figure()

# First plot: Exponential priors
ax1 = Axis(fig[1, 1], xlabel = L"\nu", ylabel = "Density")
lines!(ax1, xx, y1, label = "Exponential(0.1)", color = cols[1])
lines!(ax1, xx, y2, label = "Exponential(1)", color = cols[2])
lines!(ax1, xx, y3, label = "Exponential(5)", color = cols[3])

# Second plot: Line-only Density of σᵧₓ / σₓₓ
ax2 = Axis(fig[1, 2], xlabel = L"\sigma_{yx} / \sigma_{xx}", ylabel = "Density")
density!(ax2, data_exp12, label = "Exponential(0.1)", color = :transparent, strokecolor = cols[1], strokewidth = 1.5)
density!(ax2, data_exp22, label = "Exponential(1)", color = :transparent, strokecolor = cols[2], strokewidth = 1.5)
density!(ax2, data_exp32, label = "Exponential(5)", color = :transparent, strokecolor = cols[3], strokewidth = 1.5)
density!(ax2, data_ν3, label = L"\nu = 3", color = :transparent, strokecolor = cols[5], strokewidth = 1.5, linestyle = :dash)
density!(ax2, data_ν5, label = L"\nu = 5", color = :transparent, strokecolor = cols[6], strokewidth = 1.5, linestyle = :dash) 
xlims!(ax2, -3, 3)

# Third plot: Implied prior on \sigma_{y|x}
ax3 = Axis(fig[1, 3], xlabel = L"\sigma_{y \mid x}", ylabel = "Density")
density!(ax3, data_exp11, label = "Exponential(0.1)", color = :transparent, strokecolor = cols[1], strokewidth = 1.5)
density!(ax3, data_exp21, label = "Exponential(1)", color = :transparent, strokecolor = cols[2], strokewidth = 1.5)
density!(ax3, data_exp31, label = "Exponential(5)", color = :transparent, strokecolor = cols[3], strokewidth = 1.5)
lines!(ax3, InverseGamma(3/2, 1/2), color = cols[5], label = L"\nu = 3", linestyle = :dash)
lines!(ax3, InverseGamma(5/2, 1/2), color = cols[6], label = L"\nu = 5", linestyle = :dash)
xlims!(ax3, -1/4, 3)

# Add a shared legend below the plots
Legend(fig[2, 1:3], ax2, orientation = :horizontal)

# Adjust layout
fig[1, 1:2] = GridLayout(padding = (10, 10, 10, 40)) # Add space for the legend below

# Save the figure
save("Implied_prior_with_exponential.pdf", fig, size = (900, 400))
