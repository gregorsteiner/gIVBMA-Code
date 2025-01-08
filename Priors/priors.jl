using CairoMakie
using Distributions, Random, LinearAlgebra, KernelDensity

# Define the Exponential priors
xx = 2:0.01:5
y1 = pdf.(Exponential(0.1), xx .- 2)
y2 = pdf.(Exponential(1), xx .- 2)
y3 = pdf.(Exponential(5), xx .- 2)

# Function to sample σᵧₓ / σₓₓ
function sample_σ12(λ, m = 100000)
    ν = rand(Exponential(λ), m) .+ 2
    Σ = map(x -> rand(InverseWishart(x, [1 0; 0 1])), ν)
    return map(x -> x[1, 2] / x[2, 2], Σ)
end

# Generate data for the second plot
data_exp1 = sample_σ12(0.1)
data_exp2 = sample_σ12(1)
data_exp3 = sample_σ12(5)
data_ν3 = map(x -> x[1, 2] / x[2, 2], rand(InverseWishart(3, [1 0; 0 1]), 100000))
data_ν5 = map(x -> x[1, 2] / x[2, 2], rand(InverseWishart(5, [1 0; 0 1]), 100000))

# Helper function for computing density
function density_line(data)
    data = data[-5 .< data .< 5] # exclude extreme values from the plot
    kde = KernelDensity.kde(data)
    return kde.x, kde.density
end

# Create the figure and axis
fig = Figure()

# set up colour palette
cols = Makie.wong_colors()

# First plot: Exponential priors
ax1 = Axis(fig[1, 1], xlabel = "ν", ylabel = "Density")
lines!(ax1, xx, y1, label = "Exponential(0.1)", color = cols[1])
lines!(ax1, xx, y2, label = "Exponential(1)", color = cols[2])
lines!(ax1, xx, y3, label = "Exponential(5)", color = cols[3])

# Second plot: Line-only Density of σᵧₓ / σₓₓ
ax2 = Axis(fig[1, 2], xlabel = "σᵧₓ / σₓₓ", ylabel = "Density")
x1, d1 = density_line(data_exp1)
x2, d2 = density_line(data_exp2)
x3, d3 = density_line(data_exp3)
x4, d4 = density_line(data_ν3)
x5, d5 = density_line(data_ν5)

lines!(ax2, x1, d1, label = "Exponential(0.1)", color = cols[1])
lines!(ax2, x2, d2, label = "Exponential(1)", color = cols[2])
lines!(ax2, x3, d3, label = "Exponential(5)", color = cols[3])
lines!(ax2, x4, d4, label = "ν = 3", color = cols[5], linestyle = :dash)
lines!(ax2, x5, d5, label = "ν = 5", color = cols[6], linestyle = :dash)

# Add a shared legend below the plots
Legend(fig[2, 1:2], ax2, orientation = :horizontal)

# Adjust layout
fig[1, 1:2] = GridLayout(padding = (10, 10, 10, 40)) # Add space for the legend below

# Save the figure
save("Implied_prior_with_exponential.pdf", fig)

