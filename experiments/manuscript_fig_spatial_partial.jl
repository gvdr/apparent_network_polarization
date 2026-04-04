using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CairoMakie, LaTeXStrings
using Distributions, Random, Statistics, LinearAlgebra

"""
Partial Barberá-style robustness check.

We simulate the spatial-following model with user-specific intercepts and then
fit, for each observed user, a conditional MAP estimate of (theta_i, gamma_i)
holding the elite positions and popularity terms fixed at their known values.

This is not a full joint Bayesian fit of the original model. It is a
computationally light robustness check for the reviewer objection that
user-specific gregariousness terms may absorb the activity effect.
"""

logistic(z) = 1.0 / (1.0 + exp(-z))

function kde_curve(x::AbstractVector{<:Real}; npts::Int=400)
    xs = collect(x)
    n = length(xs)
    @assert n > 1

    σ = std(xs)
    iqr = quantile(xs, 0.75) - quantile(xs, 0.25)
    scale = min(σ, iqr / 1.34)
    if !isfinite(scale) || scale <= 0
        scale = max(σ, 1e-2)
    end
    h = 0.9 * scale * n^(-1 / 5)
    h = max(h, 0.08)

    xmin = minimum(xs) - 3h
    xmax = maximum(xs) + 3h
    grid = collect(range(xmin, xmax, length=npts))
    dens = similar(grid)

    inv_nh = 1.0 / (n * h)
    kern = Normal(0, 1)
    for (k, g) in pairs(grid)
        s = 0.0
        for xi in xs
            s += pdf(kern, (g - xi) / h)
        end
        dens[k] = inv_nh * s
    end

    return grid, dens
end

function log1pexp(z)
    if z > 0
        return z + log1p(exp(-z))
    else
        return log1p(exp(z))
    end
end

function fit_user_map(y::AbstractVector{<:Real}, elite_pos::AbstractVector{<:Real},
                      elite_pop::AbstractVector{<:Real}, κ::Float64;
                      σθ::Float64=1.5, σγ::Float64=1.0,
        max_iter::Int=25)
    # sensible initialization from followed elites
    followed = findall(>(0), y)
    θ = isempty(followed) ? 0.0 : mean(elite_pos[followed])
    η0 = mean(elite_pop .- κ .* (θ .- elite_pos).^2)
    p0 = clamp(mean(y), 1e-4, 1 - 1e-4)
    γ = log(p0 / (1 - p0)) - η0

    for _ in 1:max_iter
        dη_dθ = @. -2 * κ * (θ - elite_pos)
        η = @. elite_pop + γ - κ * (θ - elite_pos)^2
        p = logistic.(η)
        w = @. p * (1 - p)

        # gradient of log-posterior
        gθ = sum((y .- p) .* dη_dθ) - θ / σθ^2
        gγ = sum(y .- p) - γ / σγ^2
        g = [gθ, gγ]

        if norm(g) < 1e-8
            break
        end

        # Hessian of log-posterior
        hθθ = sum((y .- p) .* (-2 * κ) .- w .* (dη_dθ .^ 2)) - 1 / σθ^2
        hγγ = -sum(w) - 1 / σγ^2
        hθγ = -sum(w .* dη_dθ)
        H = [hθθ hθγ; hθγ hγγ]

        step = try
            H \ g
        catch
            pinv(H) * g
        end

        # damped Newton ascent on the log-posterior
        function logpost(θv, γv)
            lp = -0.5 * (θv / σθ)^2 - 0.5 * (γv / σγ)^2
            for j in eachindex(elite_pos)
                ηj = elite_pop[j] + γv - κ * (θv - elite_pos[j])^2
                lp += y[j] * ηj - log1pexp(ηj)
            end
            return lp
        end

        cur = logpost(θ, γ)
        accepted = false
        for λ in (1.0, 0.5, 0.25, 0.1, 0.05)
            θ_new = θ - λ * step[1]
            γ_new = γ - λ * step[2]
            new = logpost(θ_new, γ_new)
            if isfinite(new) && new >= cur
                θ, γ = θ_new, γ_new
                accepted = true
                break
            end
        end
        if !accepted
            break
        end
    end

    return θ, γ
end

function simulate_follow_matrix(opinions, gammas, elite_pos, elite_pop, κ, rng)
    N = length(opinions)
    M = length(elite_pos)
    Y = zeros(Int, N, M)
    for i in 1:N, j in 1:M
        η = elite_pop[j] + gammas[i] - κ * (opinions[i] - elite_pos[j])^2
        Y[i, j] = rand(rng) < logistic(η) ? 1 : 0
    end
    return Y
end

function keep_observed(opinions, Y, rng; α::Float64=0.0, selected::Bool=true)
    if !selected
        idx = collect(1:length(opinions))
    else
        a = @. 1.0 + α * opinions^2
        p = a ./ maximum(a)
        idx = [i for i in eachindex(opinions) if rand(rng) < p[i]]
    end
    idx = [i for i in idx if sum(Y[i, :]) > 0]
    return opinions[idx], Y[idx, :], idx
end

function figure_spatial_partial()
    rng = Xoshiro(4242)

    # shared parameters
    N_pop = 2000
    M_elites = 40
    elite_pos = collect(range(-3.0, 3.0, length=M_elites))
    elite_pop = @. 1.2 - 0.08 * elite_pos^2 + 0.15 * sin(2 * elite_pos)
    κ = 1.1

    # Model A: unimodal population + activity-linked gregariousness + observation filter
    opinions_A = rand(rng, Normal(0, 1.5), N_pop)
    gammas_A = -0.8 .+ 0.75 .* (opinions_A .^ 2) .+ rand(rng, Normal(0, 0.25), N_pop)
    Y_A = simulate_follow_matrix(opinions_A, gammas_A, elite_pos, elite_pop, κ, rng)
    obs_A, Yobs_A, idx_A = keep_observed(opinions_A, Y_A, rng; α=3.0, selected=true)

    # Model P: bimodal population + activity-neutral gregariousness, no observation filter
    opinions_P = [rand(rng) < 0.5 ? rand(rng, Normal(-2.2, 0.4)) : rand(rng, Normal(2.2, 0.4))
                  for _ in 1:N_pop]
    gammas_P = fill(-0.3, N_pop)
    Y_P = simulate_follow_matrix(opinions_P, gammas_P, elite_pos, elite_pop, κ, rng)
    obs_P, Yobs_P, idx_P = keep_observed(opinions_P, Y_P, rng; α=0.0, selected=false)

    θhat_A = Float64[]
    γhat_A = Float64[]
    for i in 1:size(Yobs_A, 1)
        θ, γ = fit_user_map(vec(Yobs_A[i, :]), elite_pos, elite_pop, κ)
        push!(θhat_A, θ)
        push!(γhat_A, γ)
    end

    θhat_P = Float64[]
    for i in 1:size(Yobs_P, 1)
        θ, _ = fit_user_map(vec(Yobs_P[i, :]), elite_pos, elite_pop, κ)
        push!(θhat_P, θ)
    end

    # orient estimated scores
    if cor(θhat_A, obs_A) < 0
        θhat_A .= -θhat_A
    end
    if cor(θhat_P, obs_P) < 0
        θhat_P .= -θhat_P
    end

    x_pop_A, d_pop_A = kde_curve(opinions_A)
    x_obs_A, d_obs_A = kde_curve(obs_A)
    x_fit_A, d_fit_A = kde_curve(θhat_A)
    x_fit_P, d_fit_P = kde_curve(θhat_P)

    fig = Figure(size=(930, 320))

    ax1 = Axis(fig[1, 1],
        xlabel=L"x",
        ylabel=L"\mathrm{density}",
        title=L"\mathrm{Model\ A:\ population\ vs\ observed}")
    hist!(ax1, opinions_A, bins=50, normalization=:pdf, color=(:gray70, 0.5),
        label=L"\mathrm{population}\ f")
    hist!(ax1, obs_A, bins=50, normalization=:pdf, color=(:royalblue, 0.5),
        label=L"\mathrm{observed}\ f_{\mathrm{obs}}")
    lines!(ax1, x_pop_A, d_pop_A, color=:gray35, linewidth=2)
    lines!(ax1, x_obs_A, d_obs_A, color=:royalblue, linewidth=2.5)
    axislegend(ax1, position=:lt, framevisible=false)

    ax2 = Axis(fig[1, 2],
        xlabel=L"\hat{\theta}\ \mathrm{(conditional\ spatial\ fit)}",
        ylabel=L"\mathrm{density}",
        title=L"\mathrm{Recovered\ ideology\ distributions}")
    hist!(ax2, θhat_A, bins=50, normalization=:pdf, color=(:royalblue, 0.5),
        label=L"\mathrm{Model\ A}")
    hist!(ax2, θhat_P, bins=50, normalization=:pdf, color=(:firebrick, 0.35),
        label=L"\mathrm{Model\ P}")
    lines!(ax2, x_fit_A, d_fit_A, color=:royalblue, linewidth=2.5)
    lines!(ax2, x_fit_P, d_fit_P, color=:firebrick, linewidth=2.5)
    axislegend(ax2, position=:lt, framevisible=false)

    ax3 = Axis(fig[1, 3],
        xlabel=L"x_i\ \mathrm{(true\ opinion)}",
        ylabel=L"\hat{\theta}_i\ \mathrm{(conditional\ fit)}",
        title=L"\mathrm{Model\ A:\ recovered\ } \hat{\theta}_i")
    scatter!(ax3, obs_A, θhat_A, markersize=3, color=(:royalblue, 0.3))

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    latexdir = joinpath(@__DIR__, "..", "..", "latex", "figures")
    mkpath(outdir)
    mkpath(latexdir)

    pdf_path = joinpath(outdir, "fig_spatial_partial_fit.pdf")
    png_path = joinpath(outdir, "fig_spatial_partial_fit.png")
    latex_pdf_path = joinpath(latexdir, "fig_spatial_partial_fit.pdf")
    latex_png_path = joinpath(latexdir, "fig_spatial_partial_fit.png")

    save(png_path, fig, px_per_unit=3)
    cp(png_path, latex_png_path; force=true)

    try
        save(pdf_path, fig)
        cp(pdf_path, latex_pdf_path; force=true)
    catch err
        @warn "PDF export failed for fig_spatial_partial_fit" exception=(err, catch_backtrace())
    end

    println("Saved fig_spatial_partial_fit PNG to " * png_path)
    println("Copied manuscript figure to " * latex_png_path)
    println("Saved fig_spatial_partial_fit PDF to " * pdf_path)
end

figure_spatial_partial()
