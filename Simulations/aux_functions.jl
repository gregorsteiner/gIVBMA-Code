



"""
    Count the number of instruments implied by a model
"""
function count_zero_one(a, b)
    length(a) == length(b) || throw(ArgumentError("Vectors must have the same length"))
    return sum((a .== 0) .& (b .== 1))
end


"""
    Extract a posterior sample of the number of implied instruments from a posterior sample of the models.
    Currently only works for the fully flexible case, where all instruments can be included in both models.
"""
function extract_instruments(L, M)
    iters = size(L, 2)
    N_Z = Vector{Int64}(undef, iters)
    for i in 1:iters
        N_Z[i] = count_zero_one(L[:, i], M[:, i])
    end
    return N_Z
end

"""
    Compute probabilities for certain events of interest given a sample of instruments.
"""
function instrument_probabilities(N_Z, p, s)
    counts = zeros(4)
    ps  = p - s
    for x in N_Z
        if x == 0
            counts[1] += 1
        elseif 0 < x < ps
            counts[2] += 1
        elseif x == ps
            counts[3] += 1
        else # x > ps
            counts[4] += 1
        end
    end
    return (counts ./ length(N_Z))
end



