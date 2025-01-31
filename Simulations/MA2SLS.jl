
##### This file implements the MA2SLS estimator by Kuersteiner & Okui (2010, Econometrica). I translated their Matlab Code into Julia. #####

using LinearAlgebra
using JuMP, Ipopt  # For solving quadratic programs

# MA2SLS function for the 'U' case, without 'con' argument
function MA2SLS_raw(y, X1, X2, Z)
    N = size(Z, 1)

    X1 = hcat(ones(N), X1)

    Z = hcat(X1, Z)
    X = hcat(X1, X2)

    M = size(Z, 2)
    d = size(X, 2)
    d1 = size(X1, 2)

    # First stage parameter estimation
    par = ParamEst(y, X, Z, ones(d) / sqrt(d), "2SLS")

    a = par.a
    b = par.b
    B = par.B
    se = par.se
    sl = par.sl
    U = par.U
    UDN = par.UDN
    W = ones(M)

    W = SWTSLSsq(W, a, se, sl, UDN, N, d)

    # 'U' case
    W, erc = SWTSLS(W, a, b, B, se, sl, U, N, d1)

    Wp = max.(W, 0)
    Wm = max.(-W, 0)
    K = collect(1:M)
    KWp = dot(K, Wp)
    KWm = dot(K, Wm)

    # Second stage
    b, stdb, pseudoR = MA2SLS_Core(y, X, Z, W, d1)

    return b, stdb, pseudoR, KWp, KWm, erc
end

# MA2SLS_Core function
function MA2SLS_Core(y, X, Z, W, d1)
    n = size(y, 1)
    M = size(Z, 2)

    if length(W) < M
        W = vcat(W, zeros(M - length(W)))
    end

    PW = zeros(n, n)
    for j = 1:M
        Z1 = Z[:, 1:j]
        PW += W[j] * (Z1 * inv(Z1' * Z1) * Z1')
    end

    b = inv(X' * PW * X) * X' * PW * y
    u = y - X * b
    Xu = (X' * PW)' .* (u * ones(1, size(X, 2)))
    V = (Xu' * Xu) / n

    stdb = sqrt.(diag(inv(X' * PW * X / n) * V / (X' * PW * X / n) / n))

    if size(X, 2) - d1 == 1
        X = X[:, end]
        pseudoR = (X' * PW * X)^2 / ((X' * PW * PW * X) * (X' * X))
    else
        pseudoR = nothing
    end

    return b, stdb, pseudoR
end

# ParamEst function
function ParamEst(y, X, Z, l, Est)
    N = size(y, 1)

    # First-stage Mallows estimation
    M = FirstStageMallows(X, Z, l)
    ZS = Z
    Z = Z[:, 1:M]

    Pi = inv(X' * X) * (X' * Z)
    Xh = Z * Pi'
    u = X - Xh
    H = Xh' * Xh / N
    Hi = inv(H)

    if Est == "2SLS"
        beta = TSLS(y, X, Z)
    else
        error("Only '2SLS' estimation is supported")
    end

    e = y - X * beta
    se = (e' * e) / N
    ul = u * inv(H) * l
    sl = (ul' * ul) / N
    sle = (ul' * e) / N

    a = sle^2
    b = se * sl + sle^2

    if size(beta, 1) == 1
        B = 2 * (se * sl + 4 * sle^2)
    else
        B = l' * Hi * CompBn(u, e, se, Xh, H) * Hi * l
    end

    Z = ZS
    M = size(Z, 2)
    PM = Z * inv(Z' * Z) * Z'
    ul = zeros(M, N)
    ulDN = zeros(M, N)

    for i in 1:M
        Zm = Z[:, 1:i]
        Pm = Zm * inv(Zm' * Zm) * Zm'
        ul[i, :] = (PM - Pm) * X * Hi * l
        ulDN[i, :] = (I(N) - Pm) * X * Hi * l
    end

    U = ul * ul'
    UDN = ulDN * ulDN'

    return (a=a, b=b, B=B, se=se, sl=sl, U=U, UDN=UDN)
end

# SWTSLSsq function
function SWTSLSsq(W, a, se, sl, U, N, d)
    M = length(W)
    K = collect(1:M)
    G = zeros(M, M)
    for i in 1:M
        G += vcat(zeros(i-1, M), hcat(zeros(M-i+1, i-1), ones(M-i+1, M-i+1)))
    end

    s = Inf
    m = 0
    for i in d:M
        W = zeros(M)
        W[i] = 1
        KW = dot(K, W)
        WGW = W' * G * W
        sNew = (a * KW^2) / N + se * (W' * U * W - sl * (-2 * KW + WGW)) / N
        if sNew < s
            s = sNew
            m = i
        end
    end

    W = zeros(M)
    W[m] = 1
    return W
end

# SWTSLS function rewritten using JuMP for quadratic programming
# Modify the SWTSLS function to suppress solver output
using JuMP
using Ipopt  # You can replace this with another solver if needed

function SWTSLS(W, a, b, B, se, sl, U, N, d1)
    M = length(W)
    K = collect(1:M)
    G = zeros(M, M)
    for i in 1:M
        G += vcat(zeros(i-1, M), hcat(zeros(M-i+1, i-1), ones(M-i+1, M-i+1)))
    end

    H = 2 * (a * (K * K') + b * G + se * (U - sl * G)) / N
    H = (H + H') / 2
    f = (-B * K - se * sl * (ones(M) * M - 2 * K)) / N

    # Create the JuMP model and set Ipopt as the optimizer, suppressing output
    model = Model(Ipopt.Optimizer)
    set_silent(model)  # Suppresses all solver messages

    # Define the variable for W
    @variable(model, W_var[1:M])

    # Define the quadratic objective
    @objective(model, Min, 0.5 * W_var' * H * W_var + f' * W_var)

    # Define the equality constraints
    @constraint(model, sum(W_var) == 1)
    for i in 1:d1
        @constraint(model, W_var[i] == 0)
    end

    # Solve the quadratic program
    optimize!(model)

    # Check for successful optimization
    if termination_status(model) == MOI.OPTIMAL
        W_opt = value.(W_var)
        erc = 1  # exit flag, success
    else
        W_opt = W  # Return original W if optimization fails
        erc = 0  # exit flag, failure
    end

    return W_opt, erc
end


# TSLS function
function TSLS(y, X, Z)
    PW = Z * inv(Z' * Z) * Z'
    b = inv(X' * PW * X) * X' * PW * y
    return b
end

# CompBn function
function CompBn(u, e, se, f, H)
    N = length(u)
    d = size(H, 1)
    SigU = (u' * u) / N
    sue = (u' * e) / N

    iH = inv(H)

    c1 = sue' * iH * sue
    c2 = sue' * iH
    c3 = c2'

    f2 = f * c3

    B = se * SigU + 2 * d * sue * sue' + c1 * H + (f' * f2 / N) * sue' + ((f' * f2 / N) * sue')'
    return B
end

# FirstStageMallows function
function FirstStageMallows(X, Z, l)
    K = size(Z, 2)
    N = size(Z, 1)
    eK = X - Z * (X \ Z)'
    sK = (eK' * eK) / N
    d = size(X, 2)
    mal = Inf
    M = 0

    for i in d:K
        ei = X - Z[:, 1:i] * (X \ Z[:, 1:i])'
        mali = l' * (ei' * ei / N + 2 * sK * i / N) * l
        if mali < mal
            mal = mali
            M = i
        end
    end

    M = max(d, M)
    return M
end
