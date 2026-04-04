using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics
using Random

println("=== Proposition B1: Q_pop = -EI/2 ===")
println()

L = 3.0
kappa = 0.1
n_samples = 200_000
rng = Xoshiro(42)

s_values = range(0.0, 3.0, length=30)
Q_pop_vec = Float64[]
neg_ei_half_vec = Float64[]

for s in s_values
    x_samp = rand(rng, Uniform(-L, L), n_samples)
    y_samp = rand(rng, Uniform(-L, L), n_samples)
    w_xy = [(1.0 + s * xi^2) * (1.0 + s * yi^2) for (xi, yi) in zip(x_samp, y_samp)]
    w_xy ./= sum(w_xy)

    u_samp = abs.(x_samp)
    v_samp = abs.(y_samp)
    W = sum(w_xy .* (1.0 .- kappa .* abs.(u_samp .- v_samp)))
    B = sum(w_xy .* (1.0 .- kappa .* (u_samp .+ v_samp)))

    ei_val = (B - W) / (B + W)
    q_pop = W / (W + B) - 0.5

    push!(Q_pop_vec, q_pop)
    push!(neg_ei_half_vec, -ei_val / 2.0)
end

max_diff = maximum(abs.(Q_pop_vec .- neg_ei_half_vec))
println("Max |Q_pop - (-EI/2)|: " * string(round(max_diff, digits=10)))

verdict = max_diff < 0.01
if verdict
    println("PASS")
else
    println("FAIL: max difference = " * string(max_diff))
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="filter strength s", ylabel="value",
          title="Proposition B1: Q_pop vs -EI/2")
lines!(ax, collect(s_values), Q_pop_vec, label="Q_pop(g)")
lines!(ax, collect(s_values), neg_ei_half_vec, label="-EI(g)/2", linestyle=:dash)
axislegend(ax)
save(joinpath(@__DIR__, "..", "output", "theorems", "propB1_modularity_ei.png"), fig)
println("Figure saved.")
verdict
