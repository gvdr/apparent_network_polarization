using DataFrames
using CSV
using Distributions
using Statistics
using Random

function wilson_interval(k::Integer, n::Integer; z::Float64=1.96)
    if n == 0
        return (center=NaN, lo=NaN, hi=NaN)
    end
    p = k / n
    denom = 1 + z^2 / n
    center = (p + z^2 / (2n)) / denom
    radius = z * sqrt((p * (1 - p) + z^2 / (4n)) / n) / denom
    return (center=center, lo=max(0.0, center - radius), hi=min(1.0, center + radius))
end

function summarize_boolean(name::AbstractString, successes::Integer, total::Integer;
                           pass_threshold::Float64=0.95)
    ci = wilson_interval(successes, total)
    rate = successes / max(total, 1)
    verdict = total > 0 && rate >= pass_threshold
    println(name * ": " * string(successes) * "/" * string(total) *
            " success, rate=" * string(round(rate, digits=4)) *
            ", 95% CI=[" * string(round(ci.lo, digits=4)) * ", " *
            string(round(ci.hi, digits=4)) * "]")
    return (verdict=verdict, success_rate=rate, ci_lo=ci.lo, ci_hi=ci.hi)
end

function ensure_stress_outdir()
    outdir = joinpath(@__DIR__, "..", "output", "theorems", "stress")
    mkpath(outdir)
    return outdir
end

function save_stress_csv(filename::AbstractString, rows)
    outdir = ensure_stress_outdir()
    df = DataFrame(rows)
    CSV.write(joinpath(outdir, filename), df)
    return df
end

function normalized_quadratic_gaussian_density(sigma::Float64, s::Float64)
    base = Normal(0.0, sigma)
    z = 1.0 + s * sigma^2
    return x -> pdf(base, x) * (1.0 + s * x^2) / z
end

function sample_tilted_gaussian_pair_expectations(rng::AbstractRNG, sigma::Float64, alpha::Float64,
                                                  n_samples::Int)
    x = randn(rng, n_samples) .* sigma
    y = randn(rng, n_samples) .* sigma
    w = (1.0 .+ alpha .* x.^2) .* (1.0 .+ alpha .* y.^2)
    w ./= sum(w)
    u = abs.(x)
    v = abs.(y)
    return (
        E_min=sum(w .* min.(u, v)),
        E_max=sum(w .* max.(u, v)),
    )
end

function sample_tilted_uniform_pair_stats(rng::AbstractRNG, L::Float64, s::Float64, kappa::Float64,
                                          n_samples::Int)
    x = rand(rng, Uniform(-L, L), n_samples)
    y = rand(rng, Uniform(-L, L), n_samples)
    w = (1.0 .+ s .* x.^2) .* (1.0 .+ s .* y.^2)
    w ./= sum(w)
    u = abs.(x)
    v = abs.(y)
    W = sum(w .* (1.0 .- kappa .* abs.(u .- v)))
    B = sum(w .* (1.0 .- kappa .* (u .+ v)))
    return (
        W=W,
        B=B,
        EI=(B - W) / (B + W),
        Q=W / (W + B) - 0.5,
    )
end

function grid_modes(xs::AbstractVector, ys::AbstractVector)
    modes = Float64[]
    for i in 2:(length(ys) - 1)
        if ys[i] >= ys[i - 1] && ys[i] > ys[i + 1]
            push!(modes, xs[i])
        end
    end
    return modes
end

function convolved_grid(f_obs, sigma_eff::Float64; grid_points::Int=1000, integration_points::Int=500,
                        grid_range::Tuple{Float64, Float64}=(-6.0, 6.0))
    xs = collect(range(grid_range[1], grid_range[2], length=grid_points))
    ys = heteroskedastic_convolution(f_obs, x -> sigma_eff;
                                     grid_range=grid_range,
                                     grid_points=grid_points,
                                     integration_points=integration_points)
    return xs, ys
end

