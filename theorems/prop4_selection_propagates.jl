using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics
using StatsBase
using Random

println("=== Proposition 4: Selection Bias Propagates ===")
println()

sigma = 1.0
alpha = 2.0
rng = Xoshiro(42)

# f = N(0, 1), f_obs proportional to f(x) * (1 + alpha*x^2)
f = Normal(0.0, sigma)
phi = x -> 1.0 + alpha * x^2
Z = 1.0 + alpha * sigma^2

function cdf_fobs(x, sigma, alpha)
    z = x / sigma
    phi_z = pdf(Normal(), z)
    Phi_z = cdf(Normal(), z)
    second_moment_term = sigma^2 * (Phi_z - z * phi_z)
    return (Phi_z + alpha * second_moment_term) / (1.0 + alpha * sigma^2)
end

function sample_fobs(rng::AbstractRNG, f::Distribution, phi, n::Int; pool_multiplier::Int=30)
    pool_size = max(pool_multiplier * n, 50_000)
    pool = rand(rng, f, pool_size)
    weights = phi.(pool)
    idx = wsample(rng, 1:pool_size, weights, n)
    return pool[idx]
end

n_values = [100, 500, 1000, 5000, 20000]
ks_to_fobs = Float64[]
ks_to_f = Float64[]

for n in n_values
    # sample from f_obs via importance resampling from a large Gaussian pool
    samples = sample_fobs(rng, f, phi, n)

    # add consistent-estimator noise (shrinks with n)
    noise_std = 0.1 / sqrt(log(n))
    estimates = samples .+ randn(rng, n) .* noise_std

    # empirical CDF of estimates
    ecdf_est = ecdf(estimates)

    # compare to F (just N(0,1))
    ecdf_f = x -> cdf(f, x)

    # KS distance on a grid
    test_points = range(-4.0, 4.0, length=1000)
    ks_fobs = maximum(abs(ecdf_est(x) - cdf_fobs(x, sigma, alpha)) for x in test_points)
    ks_f_val = maximum(abs(ecdf_est(x) - ecdf_f(x)) for x in test_points)

    push!(ks_to_fobs, ks_fobs)
    push!(ks_to_f, ks_f_val)
    println("n = " * string(n) * ": KS to F_obs = " * string(round(ks_fobs, digits=4)) * ", KS to F = " * string(round(ks_f_val, digits=4)))
end

# F_obs distance should decrease; F distance should NOT decrease to 0
fobs_converging = ks_to_fobs[end] < ks_to_fobs[1]
f_not_converging = ks_to_f[end] > 0.02  # stays bounded away from 0

println()
println("KS to F_obs decreasing: " * string(fobs_converging))
println("KS to F stays positive: " * string(f_not_converging))

verdict = fobs_converging && f_not_converging
if verdict
    println("PASS")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="sample size n", ylabel="KS distance",
          title="Prop 4: score distribution converges to F_obs, not F",
          xscale=log10)
scatterlines!(ax, Float64.(n_values), ks_to_fobs, label="KS to F_obs")
scatterlines!(ax, Float64.(n_values), ks_to_f, label="KS to F")
axislegend(ax)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop4_selection_propagates.png"), fig)
println("Figure saved.")
verdict
