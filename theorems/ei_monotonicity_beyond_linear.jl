# Numerical support for the Proposition 3ter remark.
# Monte Carlo check that expected EI decreases under FOSD-ordered |X| for
# linear, Gaussian, and exponential kernels. Density family:
#   g_s(x) \propto exp(-x^2 / 2) * (1 + s * x^2).
# Supplementary numerical evidence, not a general monotonicity theorem.
#
# Stringency: runs over multiple seeds, reports the worst adjacent
# increase across seeds and s values.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using CairoMakie
using Statistics
using Random
using StatsBase: wsample

println("=== EI under linear / Gaussian / exponential kernels, FOSD sweep ===")
println()

function sample_g_s(rng, n, s)
    xs  = randn(rng, 4n)
    wts = [1.0 + s * x^2 for x in xs]
    wts ./= sum(wts)
    idx = wsample(rng, 1:length(xs), wts, n)
    return xs[idx]
end

h_linear(d, kappa)   = max(0.0, 1.0 - kappa * d)
h_gaussian(d, gamma) = exp(-gamma * d^2)
h_exp(d, beta)       = exp(-beta * d)

function ei_under_kernel(rng, s, h; n = 200_000)
    X = sample_g_s(rng, n, s)
    Y = sample_g_s(rng, n, s)
    U = abs.(X); V = abs.(Y)
    W = mean(h.(abs.(U .- V)))
    B = mean(h.(U .+ V))
    return (B - W) / (B + W)
end

n_mc     = 200_000
seeds    = [13, 17, 19, 23, 29]
s_values = collect(range(0.0, 3.0, length=25))
mc_tol   = 0.002

kernels = [
    ("linear (kappa=0.10)",   d -> h_linear(d,   0.10)),
    ("Gaussian (gamma=0.50)", d -> h_gaussian(d, 0.50)),
    ("exponential (beta=1.00)", d -> h_exp(d,    1.00)),
]

ei_means_per_kernel = Dict{String, Vector{Float64}}()
ei_bands_per_kernel = Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}()
verdict_per_kernel  = Dict{String, Bool}()

for (label, h) in kernels
    ei_matrix = zeros(Float64, length(s_values), length(seeds))
    for (j, seed) in enumerate(seeds)
        rng = Xoshiro(seed)
        for (i, s) in enumerate(s_values)
            ei_matrix[i, j] = ei_under_kernel(rng, s, h; n = n_mc)
        end
    end
    ei_mean = [mean(ei_matrix[i, :])       for i in 1:length(s_values)]
    ei_min  = [minimum(ei_matrix[i, :])    for i in 1:length(s_values)]
    ei_max  = [maximum(ei_matrix[i, :])    for i in 1:length(s_values)]

    largest_upward = maximum([maximum(max.(diff(ei_matrix[:, j]), 0.0))
                              for j in 1:length(seeds)])
    ok             = largest_upward <= mc_tol

    ei_means_per_kernel[label] = ei_mean
    ei_bands_per_kernel[label] = (ei_min, ei_max)
    verdict_per_kernel[label]  = ok

    println(label * ":")
    println("  EI mean range         = [" * string(round(minimum(ei_mean); digits=4)) *
            ", " * string(round(maximum(ei_mean); digits=4)) * "]")
    println("  largest upward jump   = " * string(round(largest_upward; digits=5)) *
            " over " * string(length(seeds)) * " seeds")
    println("  " * (ok ? "PASS" : "FAIL") *
            " (MC tolerance = " * string(mc_tol) * ")")
end

println()
verdict = all(values(verdict_per_kernel))
println((verdict ? "OVERALL PASS" : "OVERALL FAIL"))

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(720, 450))
ax = Axis(fig[1, 1], xlabel="tail parameter s", ylabel="expected oracle EI",
          title="EI under log-concave homophily kernels")
palette = (:royalblue, :darkorange, :forestgreen)
for (idx, (label, _)) in enumerate(kernels)
    ei_mean = ei_means_per_kernel[label]
    ei_min, ei_max = ei_bands_per_kernel[label]
    band!(ax, s_values, ei_min, ei_max; color = (palette[idx], 0.25))
    lines!(ax, s_values, ei_mean; label = label, color = palette[idx])
end
axislegend(ax; position = :rt)
save(joinpath(@__DIR__, "..", "output", "theorems", "ei_monotonicity_beyond_linear.png"), fig)
println("Figure saved.")
verdict
