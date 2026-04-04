using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics
using StatsBase
using Random

println("=== Proposition 3ter: EI-index Monotonicity (Linear Homophily) ===")
println()

L = 3.0
kappa = 0.1  # must satisfy 0 < kappa < 1/(2L) = 1/6
println("L = " * string(L) * ", kappa = " * string(kappa) * " (< 1/(2L) = " * string(round(1.0/(2.0*L), digits=4)) * ")")

n_samples = 200_000
rng = Xoshiro(42)

s_values = range(0.0, 5.0, length=40)
ei_analytic = Float64[]
ei_mc = Float64[]

for s in s_values
    # sample from Uniform(-L, L), reweight by (1 + s*x^2)
    x_samp = rand(rng, Uniform(-L, L), n_samples)
    y_samp = rand(rng, Uniform(-L, L), n_samples)
    w_xy = [(1.0 + s * xi^2) * (1.0 + s * yi^2) for (xi, yi) in zip(x_samp, y_samp)]
    w_xy ./= sum(w_xy)

    u = abs.(x_samp)
    v = abs.(y_samp)
    E_min = sum(w_xy .* min.(u, v))
    E_max = sum(w_xy .* max.(u, v))

    # analytic formula from Prop 3ter
    ei_a = -kappa * E_min / (1.0 - kappa * E_max)
    push!(ei_analytic, ei_a)

    # MC: compute W(g) and B(g) directly
    # same-sign pairs: distance = |u - v|, p = rho*(1 - kappa*|u-v|)
    # opposite-sign pairs: distance = u + v, p = rho*(1 - kappa*(u+v))
    # (using rho = 1 for the ratio)
    W = sum(w_xy .* (1.0 .- kappa .* abs.(u .- v)))
    B = sum(w_xy .* (1.0 .- kappa .* (u .+ v)))
    ei_m = (B - W) / (B + W)
    push!(ei_mc, ei_m)
end

# check monotonicity (should decrease as s increases)
mono_analytic = all(diff(ei_analytic) .<= 0.005)  # small MC tolerance
mono_mc = all(diff(ei_mc) .<= 0.005)

# check analytic and MC agree
max_diff = maximum(abs.(ei_analytic .- ei_mc))

println("EI analytic monotonically decreasing: " * string(mono_analytic))
println("EI Monte Carlo monotonically decreasing: " * string(mono_mc))
println("Max |analytic - MC|: " * string(round(max_diff, digits=6)))
println("EI range: [" * string(round(ei_analytic[end], digits=4)) * ", " * string(round(ei_analytic[1], digits=4)) * "]")

verdict = mono_analytic && mono_mc && max_diff < 0.01
if verdict
    println("PASS")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="filter strength s", ylabel="EI(g)",
          title="Prop 3ter: EI decreases under tail thickening")
lines!(ax, collect(s_values), ei_analytic, label="analytic formula")
lines!(ax, collect(s_values), ei_mc, label="MC (W-B)/(W+B)", linestyle=:dash)
axislegend(ax)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop3ter_ei_monotonicity.png"), fig)
println("Figure saved.")
verdict
