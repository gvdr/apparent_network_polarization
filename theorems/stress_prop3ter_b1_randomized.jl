using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Random
using Statistics

include(joinpath(@__DIR__, "stress_utils.jl"))

println("=== Stress: Proposition 3ter / B1 Randomized Linear-Homophily Tests ===")
println()

function main()
rng = Xoshiro(42)
n_trials = 40
n_samples = 120_000
s_grid = [0.0, 0.5, 1.0, 2.0, 4.0]
identity_tol = 5e-3
mono_tol = 5e-3

rows = NamedTuple[]
successes = 0

for trial in 1:n_trials
    L = rand(rng, Uniform(1.5, 4.0))
    kappa_max = 1.0 / (2.0 * L)
    kappa = rand(rng, Uniform(0.05 * kappa_max, 0.9 * kappa_max))

    eis = Float64[]
    qs = Float64[]
    max_identity_err = 0.0

    for s in s_grid
        stats = sample_tilted_uniform_pair_stats(rng, L, s, kappa, n_samples)
        push!(eis, stats.EI)
        push!(qs, stats.Q)
        max_identity_err = max(max_identity_err, abs(stats.Q + stats.EI / 2.0))
    end

    mono_ei = all(diff(eis) .<= mono_tol)
    identity_ok = max_identity_err < identity_tol
    success = mono_ei && identity_ok
    successes += success ? 1 : 0

    push!(rows, (
        trial=trial, L=L, kappa=kappa,
        max_identity_err=max_identity_err,
        min_ei_increment=maximum(diff(eis)),
        success=success,
    ))
end

df = save_stress_csv("stress_prop3ter_b1_randomized.csv", rows)
summary = summarize_boolean("Prop 3ter/B1 randomized tests", successes, n_trials; pass_threshold=0.98)
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
