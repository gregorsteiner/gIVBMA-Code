
using SpecialFunctions
using CairoMakie, LaTeXStrings

# model prior prob. mass function (log-scale)
function model_prior(x, k, a = 1, m = k/2)
    b = (k - m) / m 
    kj = sum(x)
    
    lg(x) = SpecialFunctions.loggamma(x)
    res = lg(a+b) - (lg(a) + lg(b)) + lg(a+kj) + lg(b+k-kj) - lg(a+b+k)
    return res
end


# function that counts the number of valid instruments in a model
function count_zero_one(a, b)
    length(a) == length(b) || throw(ArgumentError("Vectors must have the same length"))
    return sum((a .== 0) .& (b .== 1))
end

# helper function that creates all possible models
function all_binary_vectors(p::Int)
    [digits(i, base = 2, pad=p) for i in 0:2^p-1]
end

# prior probability mass function for valid instruments (log-scale)
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


# Illustrate in plot
ps = [5, 10, 15, 20]
fig = Figure(size = (1000, 800))

for (i, p) in enumerate(ps)
    row = (i - 1) ÷ 2 + 1  # 1,1,2,2
    col = (i - 1) % 2 + 1  # 1,2,1,2
    prob = [instrument_prior(n_z, p) for n_z in 0:p]
    ax = Axis(fig[row, col], xlabel = "Number of valid instruments", 
              ylabel = "Prior Probability", title = "p = $p",
              xgridvisible = false, ygridvisible = false)
    #hidespines!(ax, :t, :r)
    barplot!(ax, 0:p, prob, alpha = 0.7, color = :steelblue4)
end

display(fig)

save("Implied_instrument_prior.pdf", fig)
