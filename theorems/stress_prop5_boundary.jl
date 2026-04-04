using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Random
using Statistics

include(joinpath(@__DIR__, "stress_utils.jl"))

println("=== Stress: Proposition 5 Small-Noise Boundary Tests ===")
println()

function main()
rng = Xoshiro(42)
n_trials = 24
grid_points_list = [800, 1200]
integration_points = 500
sigma_eff_candidates = [0.4, 0.3, 0.2, 0.1, 0.05]

rows = NamedTuple[]
successes = 0

for trial in 1:n_trials
    sigma = rand(rng, Uniform(0.8, 1.4))
    threshold = 1.0 / (2.0 * sigma^2)
    s = threshold + rand(rng, Uniform(0.08, 0.35))
    f_obs = normalized_quadratic_gaussian_density(sigma, s)
    modes = find_modes(f_obs; grid_range=(-6.0 * sigma, 6.0 * sigma), grid_points=20_000)
    xstar = modes[end]
    delta = f_obs(xstar) - f_obs(0.0)

    certified = false
    sigma_eff = sigma_eff_candidates[end]
    max_mode_count = 0
    min_margin = Inf

    for cand in sigma_eff_candidates
        local_ok = true
        local_mode_count = typemax(Int)
        local_margin = Inf
        for grid_points in grid_points_list
            xs, ys = convolved_grid(f_obs, cand; grid_points=grid_points,
                                    integration_points=integration_points,
                                    grid_range=(-6.0 * sigma, 6.0 * sigma))
            sup_err = maximum(abs.(ys .- [f_obs(x) for x in xs]))
            if sup_err >= 2.0 * delta / 3.0
                local_ok = false
                break
            end
            modes_conv = grid_modes(xs, ys)
            idx0 = argmin(abs.(xs))
            idxm = argmin(abs.(xs .- xstar))
            local_mode_count = min(local_mode_count, length(modes_conv))
            local_margin = min(local_margin, ys[idxm] - ys[idx0])
        end
        if local_ok
            certified = true
            sigma_eff = cand
            max_mode_count = local_mode_count
            min_margin = local_margin
            break
        end
    end

    success = certified && max_mode_count >= 2 && min_margin > 0
    successes += success ? 1 : 0
    push!(rows, (
        trial=trial, sigma=sigma, s=s, threshold=threshold, delta=delta,
        sigma_eff=sigma_eff, certified=certified,
        min_mode_count=max_mode_count, min_margin=min_margin,
        success=success,
    ))
end

df = save_stress_csv("stress_prop5_boundary.csv", rows)
summary = summarize_boolean("Prop 5 small-noise preservation", successes, n_trials; pass_threshold=0.95)
verdict = summary.verdict
println("Saved " * string(nrow(df)) * " rows.")
if verdict
    println("PASS")
else
    println("FAIL")
end
verdict
end

main()
