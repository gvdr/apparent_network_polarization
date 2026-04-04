using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics
using Random
using StatsBase

println("=== Proposition 3bis: Oracle Sign-Partition Separation ===")
println()

n_samples = 200_000
rng = Xoshiro(42)
beta_values = [0.5, 1.0, 2.0, 5.0]
alpha_values = range(0.0, 5.0, length=20)

all_pass = true

for beta in beta_values
    S_values = Float64[]
    for alpha in alpha_values
        x = randn(rng, n_samples)
        weights = [1.0 + alpha * xi^2 for xi in x]
        weights ./= sum(weights)

        # weighted E[min(|X|, |Y|)] by resampling
        # S(g) = 2*beta * E_g[min(U,V)] where U,V iid from |X|~g
        u = abs.(x)
        # resample according to weights
        idx = wsample(rng, 1:n_samples, weights, n_samples)
        u_samp = u[idx]
        idx2 = wsample(rng, 1:n_samples, weights, n_samples)
        v_samp = u[idx2]
        E_min = mean(min.(u_samp, v_samp))
        S = 2.0 * beta * E_min
        push!(S_values, S)
    end

    # check monotonicity (allow small MC noise)
    local monotone = all(diff(S_values) .>= -0.05)
    println("beta = " * string(beta) * ": S range [" * string(round(S_values[1], digits=3)) * ", " * string(round(S_values[end], digits=3)) * "] monotone = " * string(monotone))
    if !monotone
        global all_pass = false
    end
end

verdict = all_pass
if verdict
    println("PASS")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="activity curvature alpha", ylabel="S(g)",
          title="Oracle separation vs tail weight")
rng2 = Xoshiro(42)
for beta in beta_values
    S_values = Float64[]
    for alpha in alpha_values
        x = randn(rng2, n_samples)
        weights = [1.0 + alpha * xi^2 for xi in x]
        weights ./= sum(weights)
        u = abs.(x)
        idx = wsample(rng2, 1:n_samples, weights, n_samples)
        u_samp = u[idx]
        idx2 = wsample(rng2, 1:n_samples, weights, n_samples)
        v_samp = u[idx2]
        push!(S_values, 2.0 * beta * mean(min.(u_samp, v_samp)))
    end
    lines!(ax, collect(alpha_values), S_values, label="beta=" * string(beta))
end
axislegend(ax)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop3bis_oracle_separation.png"), fig)
println("Figure saved.")
verdict
