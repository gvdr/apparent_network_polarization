using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Statistics

include(joinpath(@__DIR__, "stress_utils.jl"))

println("=== Stress: Proposition 3 Boundary Sweep ===")
println()

function main()
sigmas = [0.6, 1.0, 1.4, 2.0]
epsilons = [0.2, 0.1, 0.05, 0.02]
grid_points_list = [5000, 10000, 20000]

rows = NamedTuple[]
successes = 0
total = 0

for sigma in sigmas, eps in epsilons, side in (:below, :above), grid_points in grid_points_list
    threshold = 1.0 / (2.0 * sigma^2)
    s = side === :below ? threshold - eps : threshold + eps
    density = normalized_quadratic_gaussian_density(sigma, s)
    modes = find_modes(density; grid_range=(-6.0 * sigma, 6.0 * sigma), grid_points=grid_points)
    expected_modes = side === :below ? 1 : 2
    ok = length(modes) == expected_modes
    successes += ok ? 1 : 0
    total += 1
    push!(rows, (
        sigma=sigma, epsilon=eps, side=String(side), grid_points=grid_points,
        threshold=threshold, s=s, observed_modes=length(modes),
        expected_modes=expected_modes, success=ok,
    ))
end

df = save_stress_csv("stress_prop3_boundary.csv", rows)
summary = summarize_boolean("Prop 3 boundary classification", successes, total; pass_threshold=1.0)
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
