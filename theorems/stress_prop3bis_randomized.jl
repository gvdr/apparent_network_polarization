using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Random
using Statistics

include(joinpath(@__DIR__, "stress_utils.jl"))

println("=== Stress: Proposition 3bis Randomized Admissible Tests ===")
println()

function main()
rng = Xoshiro(42)
n_trials = 60
n_samples = 80_000
alpha_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
tolerance = 1e-3

rows = NamedTuple[]
successes = 0

for trial in 1:n_trials
    sigma = rand(rng, Uniform(0.7, 1.5))
    beta = rand(rng, Uniform(0.2, 3.0))
    s_values = Float64[]
    for alpha in alpha_grid
        stats = sample_tilted_gaussian_pair_expectations(rng, sigma, alpha, n_samples)
        push!(s_values, 2.0 * beta * stats.E_min)
    end
    diffs = diff(s_values)
    monotone = all(diffs .>= -tolerance)
    successes += monotone ? 1 : 0
    push!(rows, (
        trial=trial, sigma=sigma, beta=beta,
        min_increment=minimum(diffs), max_increment=maximum(diffs),
        S_alpha0=s_values[1], S_alpha_end=s_values[end],
        success=monotone,
    ))
end

df = save_stress_csv("stress_prop3bis_randomized.csv", rows)
summary = summarize_boolean("Prop 3bis randomized monotonicity", successes, n_trials; pass_threshold=0.98)
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
