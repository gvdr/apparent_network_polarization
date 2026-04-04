using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics
using Random

println("=== Proposition 2: Directional Distortion (Variance Inflation) ===")
println()

sigma = 1.0
alpha_values = range(0.0, 5.0, length=50)
n_samples = 100_000
rng = Xoshiro(42)
var_f = sigma^2
var_ratios = Float64[]

for alpha in alpha_values
    x = randn(rng, n_samples) .* sigma
    weights = [1.0 + alpha * xi^2 for xi in x]
    weights ./= sum(weights)
    wm = sum(weights .* x)
    var_obs = sum(weights .* (x .- wm).^2)
    push!(var_ratios, var_obs / var_f)
end

all_geq_one = all(var_ratios .>= 1.0 - 1e-6)
strictly_above = all(var_ratios[alpha_values .> 0.01] .> 1.0 + 1e-6)

println("Var ratio >= 1 for all alpha: " * string(all_geq_one))
println("Var ratio > 1 for alpha > 0: " * string(strictly_above))
println("Range: [" * string(round(minimum(var_ratios), digits=4)) * ", " * string(round(maximum(var_ratios), digits=4)) * "]")

verdict = all_geq_one && strictly_above
if verdict
    println("PASS")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="activity curvature alpha", ylabel="Var(f_obs) / Var(f)",
          title="Variance inflation under activity bias")
lines!(ax, collect(alpha_values), var_ratios)
hlines!(ax, [1.0], color=:red, linestyle=:dash, label="no inflation")
axislegend(ax)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop2_directional_distortion.png"), fig)
println("Figure saved.")
verdict
