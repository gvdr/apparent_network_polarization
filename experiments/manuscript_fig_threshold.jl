using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using DataFrames, Colors

function figure_threshold()
    sigma = 1.0
    s_values = [0.0, 0.2, 0.4, 0.5, 0.7, 1.0, 2.0]
    s_threshold = 0.5
    xs = range(-4.0, 4.0, length=500)
    dx = xs[2] - xs[1]

    # Use a colorblind-safe sequential palette (viridis)
    # Map s values to colors along the viridis scale
    s_min = minimum(s_values)
    s_max = maximum(s_values)
    cmap = cgrad(:viridis)

    set_aog_theme!()
    fig = Figure(size=(700, 400))
    ax = Axis(fig[1, 1],
              xlabel=L"x",
              ylabel=L"g_s(x)",
              title=L"\mathrm{Density\ family\ across\ bimodality\ threshold}")

    for s in s_values
        ys = [exp(-x^2 / (2.0 * sigma^2)) * (1.0 + s * x^2) for x in xs]
        ys ./= sum(ys) * dx

        # color from viridis, proportional to s
        t = (s - s_min) / (s_max - s_min)
        col = cmap[t]

        if s == s_threshold
            lines!(ax, collect(xs), ys, color=col, linewidth=3.5, linestyle=:dash,
                   label=L"s = 0.5\ \mathrm{(threshold)}")
        else
            lines!(ax, collect(xs), ys, color=col, linewidth=1.8,
                   label="s = " * string(s))
        end
    end

    axislegend(ax, position=:rt, framevisible=false)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "fig3_threshold_family.pdf"), fig)
    save(joinpath(outdir, "fig3_threshold_family.png"), fig, px_per_unit=3)
    println("Saved fig3_threshold_family")
end

figure_threshold()
