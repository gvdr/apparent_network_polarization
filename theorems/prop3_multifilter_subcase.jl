using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie

println("=== Proposition 3: Multi-filter Threshold Sub-case ===")
println()

sigma = 1.0
threshold = 1.0 / (2.0 * sigma^2)  # = 0.5
println("Single-filter threshold: " * string(threshold))

# K = 4 filters, each with ratio 0.15 (below 0.5), but sum = 0.6 > 0.5
K = 4
ratios = fill(0.15, K)
println("K = " * string(K) * " filters, individual ratios: " * string(ratios))
println("Sum of ratios: " * string(sum(ratios)))
println("Above threshold: " * string(sum(ratios) > threshold))

# Product density: f(x) * prod_k (1 + r_k * x^2)
density = x -> begin
    val = exp(-x^2 / (2.0 * sigma^2))
    for r in ratios
        val *= (1.0 + r * x^2)
    end
    val
end

modes = find_modes(density; grid_range=(-6.0, 6.0), grid_points=20000)
println("Number of modes: " * string(length(modes)))

# Also verify each individual filter is below threshold
individual_bimodal = false
for (k, r) in enumerate(ratios)
    d_single = x -> exp(-x^2 / (2.0 * sigma^2)) * (1.0 + r * x^2)
    m = find_modes(d_single; grid_range=(-6.0, 6.0), grid_points=20000)
    if length(m) >= 2
        global individual_bimodal = true
        println("WARNING: individual filter " * string(k) * " is bimodal!")
    end
end

verdict = length(modes) == 2 && !individual_bimodal
if verdict
    println("PASS: multi-filter product is bimodal while no individual filter is")
else
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1], xlabel="x", ylabel="density (unnormalized)",
          title="Multi-filter product density (K=" * string(K) * ", sum=" * string(sum(ratios)) * ")")
xs = collect(range(-5.0, 5.0, length=1000))
ys = [density(x) for x in xs]
lines!(ax, xs, ys)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop3_multifilter_subcase.png"), fig)
println("Figure saved.")
verdict
