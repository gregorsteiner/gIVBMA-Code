
using SpecialFunctions
using CairoMakie

# model prior prob. mass function (log-scale)
function model_prior(x, k, a = 1, m = floor(k/2))
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
function instrument_prior(p)
    x = all_binary_vectors(p)
    prob = zeros(p+1)
    for L in x
        for M in x
            n_z = count_zero_one(L, M)
            prob[n_z + 1] += exp(model_prior(L, p) + model_prior(M, p))
        end
    end
    return prob
end


# Illustrate in plot
ps = [2, 5, 10, 15]
fig = Figure(size=(1200, 800))

for (i, p) in enumerate(ps)
    prob = instrument_prior(p)
    ax = Axis(fig[1, i], xlabel="Number of Valid Instruments", ylabel="Prior Probability", 
              title="p = $p")
    barplot!(ax, 0:p, prob, color=:steelblue4, width=0.8)
end

display(fig)

