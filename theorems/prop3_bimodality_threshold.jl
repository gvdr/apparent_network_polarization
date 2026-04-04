using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics

println("=== Proposition 3: Bimodality Threshold ===")
println()

sigma = 1.0
threshold = 1.0 / (2.0 * sigma^2)
println("Theoretical threshold: s > " * string(threshold))

s_values = range(0.0, 2.0, length=200)
n_modes_vec = Int[]
vtp_vec = Float64[]

for s in s_values
    density = x -> exp(-x^2 / (2.0 * sigma^2)) * (1.0 + s * x^2)
    modes = find_modes(density; grid_range=(-5.0, 5.0), grid_points=20000)
    push!(n_modes_vec, length(modes))
    if length(modes) >= 2
        push!(vtp_vec, valley_to_peak_ratio(density, modes))
    else
        push!(vtp_vec, 1.0)
    end
end

below = s_values .< threshold .- 0.01
above = s_values .> threshold .+ 0.01
all_unimodal_below = all(n_modes_vec[below] .== 1)
all_bimodal_above = all(n_modes_vec[above] .== 2)

println("Below threshold: all unimodal = " * string(all_unimodal_below))
println("Above threshold: all bimodal = " * string(all_bimodal_above))

verdict = all_unimodal_below && all_bimodal_above
if verdict
    println("PASS")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1], xlabel="filter strength s", ylabel="number of modes",
           title="Mode count vs filter strength")
scatter!(ax1, collect(s_values), n_modes_vec, markersize=3)
vlines!(ax1, [threshold], color=:red, linestyle=:dash, label="threshold")
axislegend(ax1)
ax2 = Axis(fig[1, 2], xlabel="filter strength s", ylabel="valley/peak ratio",
           title="Valley-to-peak ratio")
lines!(ax2, collect(s_values), vtp_vec)
vlines!(ax2, [threshold], color=:red, linestyle=:dash)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop3_bimodality_threshold.png"), fig)
println("Figure saved.")
verdict
