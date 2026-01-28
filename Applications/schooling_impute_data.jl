
using DataFrames, CSV, Random, Statistics, LinearAlgebra, Turing

##### Load and prepare data #####
d = CSV.read("card.csv", DataFrame, missingstring = "NA")[:, Not(1)]

d.expersq = d.exper.^2

covs = ["exper", "expersq", "nearc2", "nearc4", "momdad14", "sinmom14", "step14", "black", "south", "smsa", "married", "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669", "fatheduc", "motheduc"]
d = d[:, [["lwage", "educ"]; covs]]

# drop missing values in any column other than parental education
filter!(row -> !any(ismissing, row[1:(end-2)]), d)

# plot non-missing values
using StatsPlots
histogram(d.fatheduc, label = "father", alpha = 0.7, normalize=:pdf)
histogram!(d.motheduc, label = "mother", alpha = 0.7, normalize=:pdf)

##### impute parental education #####

miss_fath = ismissing.(d.fatheduc)
miss_moth = ismissing.(d.motheduc)

# check overdispersion
mean(d.fatheduc[.!miss_fath]), var(d.fatheduc[.!miss_fath])
mean(d.motheduc[.!miss_moth]), var(d.motheduc[.!miss_moth])

# fit Bayesian Poisson regression
@model function BayesPois(X, y)
    n, p = size(X)

    α ~ Normal(0, sqrt(10))
    β ~ MvNormal(zeros(p), 10 * I)
    for i in 1:n
        θ = α + dot(X[i, :], β)
        y[i] ~ Poisson(exp(θ))
    end
end

# function that draws predictions from the posterior predictive
function draw_predictions(y_train, X_train, X_test; iters = 500)
    model = BayesPois(X_train, y_train)
    chn = sample(model, NUTS(), iters; progress = true)

    alpha_values = Array(chn[:α])
    beta_values = Array(group(chn, :β))

    n_test, _ = size(X_test)
    preds = zeros(n_test)
    for i in 1:n_test
        idx = sample(1:iters)
        θ = alpha_values[idx, 1] + dot(X_test[i, :], beta_values[idx, :])
        preds[i] = rand(Poisson(exp(θ)))
    end
    return preds
end

# select relevant covariates for the imputation
covs_imp = ["nearc2", "nearc4", "momdad14", "black", "south", "smsa", "reg662", "reg663", "reg664", "reg665", "reg666", "reg667", "reg668", "reg669"]

# predict missing observations using the logic above
Random.seed!(42)
d.fatheduc[miss_fath] = draw_predictions(d.fatheduc[.!miss_fath], Matrix{Int64}(d[.!miss_fath, covs_imp]), Matrix{Int64}(d[miss_fath, covs_imp]))
d.motheduc[miss_moth] = draw_predictions(d.motheduc[.!miss_moth], Matrix{Int64}(d[.!miss_moth, covs_imp]), Matrix{Int64}(d[miss_moth, covs_imp]))

# Create plots to compare imputed and observed values
using CairoMakie

# Create figure with two subplots
fig = Figure()

# Father's education histogram
ax1 = Axis(fig[1, 1], 
    xlabel = "Father's years of education", 
    ylabel = "Frequency",
    title = "")
hist!(ax1, d.fatheduc[.!miss_fath], 
    normalization = :pdf, alpha = 0.6,
    label = "Observed")
hist!(ax1, d.fatheduc[miss_fath],  
    normalization = :pdf, alpha = 0.6,
    label = "Imputed")

# Mother's education histogram
ax2 = Axis(fig[1, 2], 
    xlabel = "Mother's years of education", 
    ylabel = "",
    title = "")
hist!(ax2, d.motheduc[.!miss_moth], 
    normalization = :pdf, alpha = 0.6,
    label = "Observed")
hist!(ax2, d.motheduc[miss_moth], 
    normalization = :pdf, alpha = 0.6,
    label = "Imputed")

fig[2, :] = Legend(fig[2, 1], ax1, orientation = :horizontal)

save("Imputed_Parental_Education.pdf", fig)

# save dataframe with imputed values
CSV.write("card_imputed.csv", d)


