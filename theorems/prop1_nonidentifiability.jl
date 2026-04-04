using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics

println("=== Proposition 1: Non-identifiability ===")
println()

# Choose an observed density: a bimodal mixture
f_obs = x -> 0.5 * pdf(Normal(-2.0, 0.8), x) + 0.5 * pdf(Normal(2.0, 0.8), x)

# Decomposition 1: f_obs IS the population, phi is flat
f1 = f_obs
phi1 = x -> 1.0

# Decomposition 2: population is unimodal N(0, 2), phi creates the bimodality
f2 = x -> pdf(Normal(0.0, 2.0), x)
# phi2 must satisfy: f_obs(x) = f2(x) * phi2(x) / Z
# so phi2(x) = f_obs(x) / f2(x) (up to normalization)
phi2_unnorm = x -> f_obs(x) / max(f2(x), 1e-30)

# Evaluate on a grid and check both decompositions produce the same f_obs
xs = collect(range(-6.0, 6.0, length=1000))
fobs_vals = [f_obs(x) for x in xs]

# Decomposition 1: f1 * phi1 / Z1
recon1 = [f1(x) * phi1(x) for x in xs]
Z1 = sum(recon1) * (xs[2] - xs[1])
recon1 ./= Z1

# Decomposition 2: f2 * phi2 / Z2
recon2 = [f2(x) * phi2_unnorm(x) for x in xs]
Z2 = sum(recon2) * (xs[2] - xs[1])
recon2 ./= Z2

# Normalize f_obs for comparison
fobs_norm = fobs_vals ./ (sum(fobs_vals) * (xs[2] - xs[1]))

sup_norm_1 = maximum(abs.(recon1 .- fobs_norm))
sup_norm_2 = maximum(abs.(recon2 .- fobs_norm))

println("Decomposition 1 (f=f_obs, phi=1): sup-norm error = " * string(round(sup_norm_1, digits=8)))
println("Decomposition 2 (f=N(0,2), phi=f_obs/f): sup-norm error = " * string(round(sup_norm_2, digits=8)))

# Verify the two population distributions are genuinely different
f1_vals = [f1(x) for x in xs]
f2_vals = [f2(x) for x in xs]
pop_diff = maximum(abs.(f1_vals .- f2_vals))
println("Population distributions differ: max|f1-f2| = " * string(round(pop_diff, digits=4)))

verdict = sup_norm_1 < 0.01 && sup_norm_2 < 0.01 && pop_diff > 0.01
if verdict
    println("PASS: two distinct decompositions produce the same f_obs")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="density",
           title="Two decompositions of f_obs")
lines!(ax1, xs, fobs_norm, label="f_obs (target)", linewidth=2)
lines!(ax1, xs, recon1, label="f1*phi1/Z1", linestyle=:dash)
lines!(ax1, xs, recon2, label="f2*phi2/Z2", linestyle=:dot)
axislegend(ax1)

ax2 = Axis(fig[1, 2], xlabel="x", ylabel="density",
           title="The two population densities")
lines!(ax2, xs, f1_vals ./ (sum(f1_vals) * (xs[2] - xs[1])), label="f1 (bimodal)")
lines!(ax2, xs, f2_vals ./ (sum(f2_vals) * (xs[2] - xs[1])), label="f2 (unimodal)")
axislegend(ax2)

save(joinpath(@__DIR__, "..", "output", "theorems", "prop1_nonidentifiability.png"), fig)
println("Figure saved.")
verdict
