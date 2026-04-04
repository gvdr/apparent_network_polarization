using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions
using CairoMakie
using Statistics

println("=== Proposition 5: Noise Preserves Bimodality ===")
println()

sigma = 1.0
s = 1.5  # well above threshold 0.5

# bimodal f_obs
f_obs = x -> pdf(Normal(0.0, sigma), x) * (1.0 + s * x^2) / (1.0 + s * sigma^2)
modes = find_modes(f_obs; grid_range=(-5.0, 5.0), grid_points=20000)
println("f_obs modes: " * string(length(modes)) * " at " * string(round.(modes, digits=3)))

valley_depth = f_obs(modes[end]) - f_obs(0.0)
println("Valley depth Delta = f_obs(x*) - f_obs(0) = " * string(round(valley_depth, digits=4)))

# Part 1: Sanity check with small noise
println()
println("--- Part 1: Sanity check ---")
candidate_sigmas = [0.3, 0.2, 0.1, 0.05, 0.02]
function convolve_with_sigma(f_obs, sigma_eff)
    grid = collect(range(-6.0, 6.0, length=1000))
    convolved = heteroskedastic_convolution(f_obs, x -> sigma_eff;
        grid_range=(-6.0, 6.0), grid_points=1000, integration_points=500)
    sup_err = maximum(abs.(convolved .- [f_obs(x) for x in grid]))
    return grid, convolved, sup_err
end

function find_grid_modes(xs, ys)
    mode_idx = Int[]
    for i in 2:(length(ys)-1)
        if ys[i] >= ys[i-1] && ys[i] > ys[i+1]
            push!(mode_idx, i)
        end
    end
    return xs[mode_idx]
end

function find_small_noise_certificate(f_obs, candidate_sigmas, valley_depth)
    sigma_eff_small = candidate_sigmas[end]
    grid, convolved, sup_err = convolve_with_sigma(f_obs, sigma_eff_small)
    bound_holds = false

    for candidate in candidate_sigmas
        trial_grid, trial_convolved, trial_sup_err = convolve_with_sigma(f_obs, candidate)
        if trial_sup_err < (2.0 * valley_depth / 3.0)
            sigma_eff_small = candidate
            grid = trial_grid
            convolved = trial_convolved
            sup_err = trial_sup_err
            bound_holds = true
            break
        end
    end

    return sigma_eff_small, grid, convolved, sup_err, bound_holds
end

sigma_eff_small, grid, convolved, sup_err, bound_holds =
    find_small_noise_certificate(f_obs, candidate_sigmas, valley_depth)

if bound_holds
    println("Found sigma_eff with uniform bound: " * string(sigma_eff_small))
    println("sup|f_hat - f_obs| = " * string(round(sup_err, digits=6)) *
            " < 2Delta/3 = " * string(round(2.0 * valley_depth / 3.0, digits=6)))
end

# find modes of convolved density
conv_modes = find_grid_modes(grid, convolved)
println("Convolved density modes (sigma_eff=" * string(sigma_eff_small) * "): " * string(length(conv_modes)))

# check f_hat(x*) > f_hat(0)
idx_zero = argmin(abs.(grid))
idx_mode = argmin(abs.(grid .- modes[end]))
println("f_hat(0) = " * string(round(convolved[idx_zero], digits=6)))
println("f_hat(x*) = " * string(round(convolved[idx_mode], digits=6)))
sanity_pass = bound_holds && convolved[idx_mode] > convolved[idx_zero]
println("f_hat(x*) > f_hat(0): " * string(sanity_pass))

# Part 2: Exploratory sweep
println()
println("--- Part 2: Exploratory sweep ---")
sigma_eff_values = range(0.05, 3.0, length=40)
n_modes_convolved = Int[]

for se in sigma_eff_values
    noise_fn = x -> se
    conv = heteroskedastic_convolution(f_obs, noise_fn;
        grid_range=(-6.0, 6.0), grid_points=500, integration_points=300)
    g = collect(range(-6.0, 6.0, length=500))
    m = find_grid_modes(g, conv)
    push!(n_modes_convolved, length(m))
end

# find transition
transition_idx = findfirst(==(1), n_modes_convolved)
if !isnothing(transition_idx)
    println("Bimodality lost at sigma_eff ~ " * string(round(sigma_eff_values[transition_idx], digits=3)))
else
    println("Bimodality preserved across entire sweep")
end

verdict = sanity_pass
if verdict
    println()
    println("PASS")
else
    println()
    println("FAIL")
end

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(800, 400))
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="density",
           title="f_obs vs convolved (sigma_eff=" * string(sigma_eff_small) * ")")
lines!(ax1, grid, [f_obs(x) for x in grid] ./ maximum(f_obs(x) for x in grid), label="f_obs")
lines!(ax1, grid, convolved ./ maximum(convolved), label="f_hat", linestyle=:dash)
axislegend(ax1)
ax2 = Axis(fig[1, 2], xlabel="sigma_eff", ylabel="number of modes",
           title="Bimodality vs noise scale")
scatter!(ax2, collect(sigma_eff_values), n_modes_convolved, markersize=5)
save(joinpath(@__DIR__, "..", "output", "theorems", "prop5_noise_preserves_bimodality.png"), fig)
println("Figure saved.")
verdict
