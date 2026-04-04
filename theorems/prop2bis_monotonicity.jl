using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie

println("=== Proposition 2bis: Monotonicity Above Threshold ===")
println()

sigma = 1.0
threshold = 1.0 / (2.0 * sigma^2)
s_values = range(threshold + 0.02, 4.0, length=100)

rho_vec = Float64[]
separation_vec = Float64[]
delta_vec = Float64[]

for s in s_values
    density = x -> exp(-x^2 / (2.0 * sigma^2)) * (1.0 + s * x^2)
    modes = find_modes(density; grid_range=(-6.0, 6.0), grid_points=20000)

    if length(modes) >= 2
        push!(rho_vec, valley_to_peak_ratio(density, modes))
        m1, m2 = extrema(modes)
        push!(separation_vec, m2 - m1)
        # excess valley mass: integral of density - density(0) over [0, mode]
        local xs = range(0.0, m2, length=500)
        d0 = density(0.0)
        delta = sum(max(0.0, density(x) - d0) for x in xs) * (xs[2] - xs[1])
        push!(delta_vec, delta)
    end
end

rho_decreasing = all(diff(rho_vec) .< 1e-10)
sep_increasing = all(diff(separation_vec) .> -1e-10)
delta_increasing = all(diff(delta_vec) .> -1e-10)

println("rho(s) strictly decreasing: " * string(rho_decreasing))
println("Mode separation increasing: " * string(sep_increasing))
println("Excess valley mass increasing: " * string(delta_increasing))

verdict = rho_decreasing && sep_increasing && delta_increasing
if verdict
    println("PASS")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(900, 300))
ax1 = Axis(fig[1, 1], xlabel="s", ylabel="rho(s)", title="Valley-to-peak ratio")
lines!(ax1, collect(s_values)[1:length(rho_vec)], rho_vec)
ax2 = Axis(fig[1, 2], xlabel="s", ylabel="2x*(s)", title="Mode separation")
lines!(ax2, collect(s_values)[1:length(separation_vec)], separation_vec)
ax3 = Axis(fig[1, 3], xlabel="s", ylabel="Delta(s)", title="Excess valley mass")
lines!(ax3, collect(s_values)[1:length(delta_vec)], delta_vec)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop2bis_monotonicity.png"), fig)
println("Figure saved.")
verdict
