using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using Random
using Statistics
using StatsBase

include(joinpath(@__DIR__, "stress_utils.jl"))

println("=== Stress: Proposition 4 Replicated Monte Carlo ===")
println()

function main()
sigma = 1.0
alpha = 2.0
f = Normal(0.0, sigma)
phi = x -> 1.0 + alpha * x^2
rng = Xoshiro(42)
n_values = [100, 500, 1000, 5000]
n_trials = 30

function cdf_fobs(x, sigma, alpha)
    z = x / sigma
    phi_z = pdf(Normal(), z)
    Phi_z = cdf(Normal(), z)
    second_moment_term = sigma^2 * (Phi_z - z * phi_z)
    return (Phi_z + alpha * second_moment_term) / (1.0 + alpha * sigma^2)
end

function sample_fobs(rng::AbstractRNG, f::Distribution, phi, n::Int; pool_multiplier::Int=40)
    pool_size = max(pool_multiplier * n, 50_000)
    pool = rand(rng, f, pool_size)
    weights = phi.(pool)
    idx = wsample(rng, 1:pool_size, weights, n)
    return pool[idx]
end

rows = NamedTuple[]
trial_successes = 0

for trial in 1:n_trials
    trial_ok = true
    for n in n_values
        samples = sample_fobs(rng, f, phi, n)
        noise_std = 0.1 / sqrt(log(n))
        estimates = samples .+ randn(rng, n) .* noise_std
        ecdf_est = ecdf(estimates)
        test_points = range(-4.0, 4.0, length=1000)
        ks_fobs = maximum(abs(ecdf_est(x) - cdf_fobs(x, sigma, alpha)) for x in test_points)
        ks_f = maximum(abs(ecdf_est(x) - cdf(f, x)) for x in test_points)
        closer_to_fobs = ks_fobs < ks_f
        trial_ok &= closer_to_fobs
        push!(rows, (
            trial=trial, n=n, ks_fobs=ks_fobs, ks_f=ks_f,
            gap=ks_f - ks_fobs, success=closer_to_fobs,
        ))
    end
    trial_successes += trial_ok ? 1 : 0
end

df = save_stress_csv("stress_prop4_replications.csv", rows)
summary = summarize_boolean("Prop 4 replicated trials", trial_successes, n_trials; pass_threshold=0.9)
verdict = summary.verdict
println("Saved " * string(nrow(df)) * " rows.")
if verdict
    println("PASS")
else
    println("FAIL")
end
verdict
end

main()
