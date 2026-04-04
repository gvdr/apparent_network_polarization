using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using Distributions, Statistics, StatsBase, Random, LinearAlgebra, DataFrames

"""
Simulate the CA/SVD ideology estimation pipeline under Model A and Model P.

Generates a users × elites bipartite follow network, applies activity-based
selection, then runs correspondence analysis (SVD on the standardized residual
matrix) to estimate ideology scores. Compares the resulting score distributions
under the two models.
"""
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

function figure_ca_svd()
    set_aog_theme!()
    rng = Xoshiro(2024)

    # ── Shared parameters ─────────────────────────────────────────────
    N_pop    = 2000       # population size
    M_elites = 80         # number of elite accounts
    elite_range = 3.0     # elites evenly spaced on [-elite_range, elite_range]
    gamma    = 1.1        # spatial following sensitivity
    alpha    = 3.0        # activity curvature (Model A)
    sigma    = 1.5        # population std (Model A)
    mu_P     = 1.8        # mode separation (Model P)
    tau_P    = 0.6        # mode width (Model P)

    elite_positions = collect(range(-elite_range, elite_range, length=M_elites))
    elite_popularity = @. 1.2 - 0.08 * elite_positions^2 + 0.15 * sin(2 * elite_positions)

    # ── Helper: generate follow matrix and select observed users ──────
    function generate_follow_matrix(opinions, gammas, rng; apply_selection=true, alpha_val=0.0)
        N = length(opinions)
        M = length(elite_positions)
        Y = zeros(Int, N, M)
        for i in 1:N, j in 1:M
            η = elite_popularity[j] + gammas[i] - gamma * (opinions[i] - elite_positions[j])^2
            p_follow = logistic(η)
            Y[i, j] = rand(rng) < p_follow ? 1 : 0
        end

        if apply_selection
            # activity-based selection: probability of observation ∝ a(x)
            a_vals = [1.0 + alpha_val * opinions[i]^2 for i in 1:N]
            max_a = maximum(a_vals)
            obs_mask = [rand(rng) < a_vals[i] / max_a for i in 1:N]
        else
            obs_mask = trues(N)
        end

        obs_idx = findall(obs_mask)
        # drop users who follow zero elites
        active = [i for i in obs_idx if sum(Y[i, :]) > 0]
        return Y[active, :], opinions[active]
    end

    function logistic(z)
        return 1.0 / (1.0 + exp(-z))
    end

    # ── Helper: correspondence analysis (SVD on standardized residuals) ──
    function ca_scores(Y)
        N, M = size(Y)
        total = sum(Y)
        if total == 0
            return zeros(N)
        end
        r = vec(sum(Y, dims=2)) ./ total   # row masses
        c = vec(sum(Y, dims=1)) ./ total   # column masses

        # standardized residual matrix
        # S_{ij} = (Y_{ij}/total - r_i * c_j) / sqrt(r_i * c_j)
        S = zeros(N, M)
        for i in 1:N, j in 1:M
            denom = sqrt(r[i] * c[j])
            if denom > 0
                S[i, j] = (Y[i, j] / total - r[i] * c[j]) / denom
            end
        end

        F = svd(S)
        # first dimension scores (weighted by row mass)
        scores = F.U[:, 1] ./ sqrt.(r)
        return scores
    end

    # ── Model A: unimodal population + activity bias ──────────────────
    rng_A = Xoshiro(1001)
    opinions_A = rand(rng_A, Normal(0.0, sigma), N_pop)
    gammas_A = -0.8 .+ 0.75 .* (opinions_A .^ 2) .+ rand(rng_A, Normal(0, 0.25), N_pop)
    Y_A, obs_opinions_A = generate_follow_matrix(opinions_A, gammas_A, rng_A;
        apply_selection=true, alpha_val=alpha)
    scores_A = ca_scores(Y_A)
    # orient scores so they correlate positively with opinions
    if cor(scores_A, obs_opinions_A) < 0
        scores_A .= -scores_A
    end

    # ── Model P: bimodal population, no activity bias ─────────────────
    rng_P = Xoshiro(2002)
    opinions_P = [rand(rng_P) < 0.5 ?
        rand(rng_P, Normal(-mu_P, tau_P)) :
        rand(rng_P, Normal(mu_P, tau_P))
        for _ in 1:N_pop]
    gammas_P = fill(-0.3, N_pop)
    Y_P, obs_opinions_P = generate_follow_matrix(opinions_P, gammas_P, rng_P;
        apply_selection=false, alpha_val=0.0)
    scores_P = ca_scores(Y_P)
    if cor(scores_P, obs_opinions_P) < 0
        scores_P .= -scores_P
    end

    # ── Also show the population distribution for Model A ─────────────
    # (all opinions, before selection)
    pop_opinions_A = opinions_A
    x_pop_A, d_pop_A = kde_curve(pop_opinions_A)
    x_obs_A, d_obs_A = kde_curve(obs_opinions_A)

    # ── Figure: three panels ──────────────────────────────────────────
    fig = Figure(size=(900, 320))

    # Panel 1: population vs observed opinions (Model A)
    ax1 = Axis(fig[1, 1],
        xlabel = L"x",
        ylabel = L"\mathrm{density}",
        title  = L"\mathrm{Model\ A:\ opinion\ distributions}")
    hist!(ax1, pop_opinions_A, bins=50, normalization=:pdf,
        color=(:gray70, 0.5), label=L"\mathrm{population}\ f")
    hist!(ax1, obs_opinions_A, bins=50, normalization=:pdf,
        color=(:royalblue, 0.5), label=L"\mathrm{observed}\ f_{\mathrm{obs}}")
    lines!(ax1, x_pop_A, d_pop_A, color=:gray35, linewidth=2)
    lines!(ax1, x_obs_A, d_obs_A, color=:royalblue, linewidth=2.5)
    axislegend(ax1, position=:lt, framevisible=false)

    # Panel 2: CA/SVD score distributions, Model A vs Model P
    # normalize scores to unit variance for comparability
    scores_A_n = (scores_A .- mean(scores_A)) ./ std(scores_A)
    scores_P_n = (scores_P .- mean(scores_P)) ./ std(scores_P)
    x_score_A, d_score_A = kde_curve(scores_A_n)
    x_score_P, d_score_P = kde_curve(scores_P_n)

    ax2 = Axis(fig[1, 2],
        xlabel = L"\hat{\theta}\ \mathrm{(CA/SVD\ score,\ standardized)}",
        ylabel = L"\mathrm{density}",
        title  = L"\mathrm{CA/SVD\ ideology\ scores}")
    hist!(ax2, scores_A_n, bins=60, normalization=:pdf,
        color=(:royalblue, 0.5), label=L"\mathrm{Model\ A}")
    hist!(ax2, scores_P_n, bins=60, normalization=:pdf,
        color=(:firebrick, 0.3), label=L"\mathrm{Model\ P}")
    lines!(ax2, x_score_A, d_score_A, color=:royalblue, linewidth=2.5)
    lines!(ax2, x_score_P, d_score_P, color=:firebrick, linewidth=2.5)
    axislegend(ax2, position=:lt, framevisible=false)

    # Panel 3: score vs true opinion (Model A) — shows the mapping
    ax3 = Axis(fig[1, 3],
        xlabel = L"x_i\ \mathrm{(true\ opinion)}",
        ylabel = L"\hat{\theta}_i\ \mathrm{(CA/SVD\ score)}",
        title  = L"\mathrm{Model\ A:\ score\ vs\ opinion}")
    scatter!(ax3, obs_opinions_A, scores_A_n,
        markersize=3, color=(:royalblue, 0.3))

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    latexdir = joinpath(@__DIR__, "..", "..", "latex", "figures")
    mkpath(outdir)
    mkpath(latexdir)
    pdf_path = joinpath(outdir, "fig_ca_svd_scores.pdf")
    png_path = joinpath(outdir, "fig_ca_svd_scores.png")
    latex_pdf_path = joinpath(latexdir, "fig_ca_svd_scores.pdf")
    latex_png_path = joinpath(latexdir, "fig_ca_svd_scores.png")
    save(pdf_path, fig)
    save(png_path, fig, px_per_unit=3)
    cp(pdf_path, latex_pdf_path; force=true)
    cp(png_path, latex_png_path; force=true)
    println("Saved fig_ca_svd_scores to " * outdir)
end

figure_ca_svd()
