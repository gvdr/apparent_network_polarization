using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using PolarizationPipeline, Distributions, DataFrames

# ── Figure 2: Non-identifiability (Proposition 1) ──────────────────────────

function figure2_nonidentifiability()
    set_aog_theme!()

    xs = range(-6.0, 6.0, length=1000)

    # f_obs = 0.5*N(-2, 0.8) + 0.5*N(2, 0.8)
    d1 = Normal(-2.0, 0.8)
    d2 = Normal(2.0, 0.8)
    f_obs(x) = 0.5 * pdf(d1, x) + 0.5 * pdf(d2, x)

    # Decomposition 1: f1 = f_obs, phi1 = 1 (flat), Z1 = 1
    f1(x) = f_obs(x)
    phi1(x) = 1.0
    Z1 = 1.0
    f1_weighted(x) = f1(x) * phi1(x) / Z1   # == f_obs

    # Decomposition 2: f2 = N(0, 2), phi2 = f_obs / f2, renormalize
    f2_dist = Normal(0.0, 2.0)
    f2(x) = pdf(f2_dist, x)
    phi2_unnorm(x) = f2(x) > 1e-12 ? f_obs(x) / f2(x) : 0.0

    # Compute Z2 numerically (integral of f2 * phi2_unnorm)
    dx = step(xs)
    Z2 = sum(f2(x) * phi2_unnorm(x) * dx for x in xs)
    f2_weighted(x) = f2(x) * phi2_unnorm(x) / Z2

    # Evaluate on grid
    ys_fobs        = [f_obs(x)        for x in xs]
    ys_f1_weighted = [f1_weighted(x)  for x in xs]
    ys_f2_weighted = [f2_weighted(x)  for x in xs]
    ys_f1          = [f1(x)           for x in xs]
    ys_f2          = [f2(x)           for x in xs]

    xs_vec = collect(xs)

    # Build figure with two Axis panels side by side
    fig = Figure(size=(800, 350))

    # ── Left panel: three overlapping reconstructions ─────────────────────
    ax_left = Axis(fig[1, 1],
        xlabel = L"x",
        ylabel = L"f(x)",
        title  = L"\mathrm{Two\ decompositions\ of}\ \hat{f}_{\mathrm{obs}}")

    wc = Makie.wong_colors()
    # solid line for f_obs as reference
    lines!(ax_left, xs_vec, ys_fobs,
        linewidth = 2.5, color = (:black, 0.6),
        label = L"f_{\mathrm{obs}}")
    # sparse markers for each decomposition (every 25th point, offset)
    stride = 25
    idx2 = 13:stride:length(xs_vec)
    idx3 = 7:stride:length(xs_vec)
    scatter!(ax_left, xs_vec[idx2], ys_f1_weighted[idx2],
        marker = :utriangle, markersize = 9, color = wc[1],
        label = L"f_1 \cdot \phi_1 / Z_1")
    scatter!(ax_left, xs_vec[idx3], ys_f2_weighted[idx3],
        marker = :rect, markersize = 7, color = wc[2],
        label = L"f_2 \cdot \phi_2 / Z_2")

    axislegend(ax_left, position = :lt, framevisible = false)

    # ── Right panel: the two genuine population densities ─────────────────
    ax_right = Axis(fig[1, 2],
        xlabel = L"x",
        ylabel = L"f(x)",
        title  = L"\mathrm{The\ two\ population\ densities}")

    lines!(ax_right, xs_vec, ys_f1,
        linewidth = 2, linestyle = :dash, color = wc[1],
        label = L"f_1\ \mathrm{(bimodal)}")
    lines!(ax_right, xs_vec, ys_f2,
        linewidth = 2, linestyle = :solid, color = wc[2],
        label = L"f_2 = N(0,2)\ \mathrm{(unimodal)}")

    axislegend(ax_right, position = :lt, framevisible = false)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "fig2_nonidentifiability.pdf"), fig)
    save(joinpath(outdir, "fig2_nonidentifiability.png"), fig, px_per_unit = 3)
    println("Saved fig2_nonidentifiability to " * outdir)
end

# ── Figure 8: Noise preserves bimodality (Proposition 5) ──────────────────

function figure8_noise_bimodality()
    set_aog_theme!()

    # f_obs: bimodal with separation parameter s = 1.5
    s = 1.5
    d_left  = Normal(-s, 0.6)
    d_right = Normal( s, 0.6)
    f_obs_raw(x) = 0.5 * pdf(d_left, x) + 0.5 * pdf(d_right, x)

    grid_range = (-6.0, 6.0)
    xs = range(grid_range[1], grid_range[2], length=1000)
    xs_vec = collect(xs)

    ys_fobs = [f_obs_raw(x) for x in xs_vec]

    # ── Left panel: f_obs vs f_hat (convolved) for sigma_eff = 0.3 ────────
    sigma_eff = 0.3
    noise_func(x) = sigma_eff

    f_hat_vals = heteroskedastic_convolution(
        f_obs_raw, noise_func;
        grid_range       = grid_range,
        grid_points      = 1000,
        integration_points = 500,
    )

    # ── Right panel: mode count sweep ─────────────────────────────────────
    sigma_grid = range(0.05, 3.0, length=60)
    n_modes_vec = Int[]

    for sig in sigma_grid
        nf(x) = sig
        fhat = heteroskedastic_convolution(
            f_obs_raw, nf;
            grid_range         = grid_range,
            grid_points        = 500,
            integration_points = 300,
        )
        # wrap array as callable for find_modes
        zs_local = collect(range(grid_range[1], grid_range[2], length=500))
        fhat_func(x) = begin
            idx = searchsortedfirst(zs_local, x)
            idx = clamp(idx, 1, length(fhat))
            fhat[idx]
        end
        modes = find_modes(fhat_func; grid_range = grid_range, grid_points = 500)
        push!(n_modes_vec, length(modes))
    end

    sigma_vec = collect(sigma_grid)

    # ── Build figure ───────────────────────────────────────────────────────
    fig = Figure(size=(800, 350))

    # Left: overlaid density curves
    ax_left = Axis(fig[1, 1],
        xlabel = L"x",
        ylabel = L"f(x)",
        title  = L"\mathrm{Noise\ preserves\ bimodality}\ (\sigma_{\mathrm{eff}} = 0.3)")

    lines!(ax_left, xs_vec, ys_fobs,
        linewidth = 2.5, color = :black,
        label = L"f_{\mathrm{obs}}")
    lines!(ax_left, xs_vec, f_hat_vals,
        linewidth = 2.5, linestyle = :dash, color = (:orange, 0.85),
        label = latexstring("\\hat{f},\\ \\sigma_{\\mathrm{eff}} = " * string(sigma_eff)))

    axislegend(ax_left, position = :lt, framevisible = false)

    # Right: mode count vs sigma_eff
    ax_right = Axis(fig[1, 2],
        xlabel = L"\sigma_{\mathrm{eff}}",
        ylabel = L"\mathrm{Number\ of\ modes}",
        title  = L"\mathrm{Bimodality\ robustness\ to\ noise}",
        yticks = [1, 2])

    lines!(ax_right, sigma_vec, n_modes_vec,
        linewidth = 2, color = :royalblue)
    scatter!(ax_right, sigma_vec, n_modes_vec,
        markersize = 5, color = :royalblue)

    # Reference line at 2 modes
    hlines!(ax_right, [2], linestyle = :dash, color = :gray60, linewidth = 1)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "fig8_noise_bimodality.pdf"), fig)
    save(joinpath(outdir, "fig8_noise_bimodality.png"), fig, px_per_unit = 3)
    println("Saved fig8_noise_bimodality to " * outdir)
end

# ── Run ────────────────────────────────────────────────────────────────────
figure2_nonidentifiability()
figure8_noise_bimodality()
