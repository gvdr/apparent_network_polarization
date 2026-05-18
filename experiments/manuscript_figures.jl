using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings, GraphMakie, Graphs, Colors
using PolarizationPipeline, Distributions, Random, Statistics

function figure_pipeline_anatomy()
    sigma = 1.0
    alpha = 2.0
    tau = 1.0

    xs = range(-4.0, 4.0, length=500)
    dx = xs[2] - xs[1]

    # Stage functions
    f = x -> pdf(Normal(0.0, sigma), x)
    a = x -> 1.0 + alpha * x^2
    t = x -> min(1.0, (1.0 + tau * x^2) / (1.0 + tau))
    # platform curation: amplification of moderate-extreme content + content moderation truncation
    # amplification: engagement-driven reach grows with |x|
    # moderation: far tails get censored (sigmoid cutoff around |x| = 3)
    p = x -> (1.0 + 0.5 * abs(x)^1.5) * (1.0 / (1.0 + exp(3.0 * (abs(x) - 3.0))))

    # Compute densities at each stage, normalized
    y_f = [f(xi) for xi in xs]; y_f ./= sum(y_f) * dx
    y_fa = [f(xi) * a(xi) for xi in xs]; y_fa ./= sum(y_fa) * dx
    y_fat = [f(xi) * a(xi) * t(xi) for xi in xs]; y_fat ./= sum(y_fat) * dx
    y_fatp = [f(xi) * a(xi) * t(xi) * p(xi) for xi in xs]; y_fatp ./= sum(y_fatp) * dx

    # --- Generate network for graph panel ---
    N_nodes = 150
    rng = Xoshiro(42)

    # Sample opinions from the composite density via rejection
    unnorm = x -> f(x) * a(x) * t(x) * p(x)
    max_val = maximum(unnorm(xi) for xi in xs)
    opinions = Float64[]
    while length(opinions) < N_nodes
        candidate = -4.0 + 8.0 * rand(rng)
        if rand(rng) < unnorm(candidate) / max_val
            push!(opinions, candidate)
        end
    end

    # Form network
    g = SimpleGraph(N_nodes)
    rho = 0.06
    beta = 1.0
    for i in 1:N_nodes
        for j in (i+1):N_nodes
            prob = clamp(rho * exp(-beta * abs(opinions[i] - opinions[j])), 0.0, 1.0)
            if rand(rng) < prob
                add_edge!(g, i, j)
            end
        end
    end

    # Giant component
    cc = connected_components(g)
    gc_nodes = cc[argmax(length.(cc))]
    g_gc, _ = induced_subgraph(g, gc_nodes)
    gc_opinions = opinions[gc_nodes]

    # Node colors: blue-white-red diverging
    function opinion_color(x)
        t_val = clamp((x + 4.0) / 8.0, 0.0, 1.0)
        if t_val <= 0.5
            s = 2.0 * t_val
            return RGB(s, s, 1.0)
        else
            s = 2.0 * (1.0 - t_val)
            return RGB(1.0, s, s)
        end
    end
    gc_colors = [opinion_color(x) for x in gc_opinions]

    # Panel 6: KDE of observed opinions (what a researcher would estimate)
    h_bw = 0.4  # bandwidth
    y_kde = zeros(length(xs))
    n_gc = length(gc_opinions)
    inv_norm = 1.0 / (n_gc * h_bw * sqrt(2.0 * pi))
    for (i, gx) in enumerate(xs)
        acc = 0.0
        for xi in gc_opinions
            z = (gx - xi) / h_bw
            acc += exp(-0.5 * z * z)
        end
        y_kde[i] = inv_norm * acc
    end

    # --- Build the figure ---
    # Layout:
    #   Row 1: [Panel 1] → [Panel 2] → [Panel 3]
    #   Row 2: [Panel 6] ← [Panel 5] ← [Panel 4]
    # With arrow columns between panels and a vertical arrow between rows 1-2

    set_aog_theme!()
    fig = Figure(size=(1100, 800), fontsize=18)

    ylims_max = 1.1 * maximum(vcat(y_f, y_fa, y_fat, y_fatp, y_kde))
    ax_kw = (; limits=((-4, 4), (0, ylims_max)))
    ax_kw_noy = (; limits=((-4, 4), (0, ylims_max)), yticklabelsvisible=false)

    # --- Top row: panels 1, 2, 3 (left to right) ---
    # Columns: 1=panel, 2=arrow, 3=panel, 4=arrow, 5=panel

    ax1 = Axis(fig[1, 1]; xlabel=L"\mathrm{opinion}", ylabel=L"\mathrm{density}",
               title=L"\mathrm{1.\ Population\ distribution}", ax_kw...)
    lines!(ax1, collect(xs), y_f, linewidth=2, color=:steelblue)

    ax2 = Axis(fig[1, 3]; xlabel=L"\mathrm{opinion}",
               title=L"\mathrm{2.\ After\ activity\ bias}", ax_kw_noy...)
    lines!(ax2, collect(xs), y_fa, linewidth=2, color=:steelblue)

    ax3 = Axis(fig[1, 5]; xlabel=L"\mathrm{opinion}",
               title=L"\mathrm{3.\ After\ data\ collection}", ax_kw_noy...)
    lines!(ax3, collect(xs), y_fat, linewidth=2, color=:steelblue)

    # --- Bottom row: panels 4, 5, 6 (right to left) ---
    # Note: panel 4 at column 5, panel 5 at column 3, panel 6 at column 1

    ax4 = Axis(fig[3, 5]; xlabel=L"\mathrm{opinion}",
               title=L"\mathrm{4.\ After\ platform\ curation}", ax_kw_noy...)
    lines!(ax4, collect(xs), y_fatp, linewidth=2, color=:steelblue)
    modes_d = find_modes(x -> f(x) * a(x) * t(x) * p(x); grid_range=(-4.0, 4.0), grid_points=10000)
    if length(modes_d) >= 2
        vlines!(ax4, modes_d, color=:red, linestyle=:dash, linewidth=1.5)
    end

    # Panel 5: graph
    ax5 = Axis(fig[3, 3]; title=L"\mathrm{5.\ Observed\ graph}")
    hidedecorations!(ax5)
    hidespines!(ax5)
    spring_layout = GraphMakie.NetworkLayout.Spring(; seed=42)
    graphplot!(ax5, g_gc;
        layout=spring_layout,
        node_color=gc_colors,
        node_size=6,
        edge_width=0.4,
        edge_color=(:gray, 0.2))

    # Panel 6: estimated opinion distribution (KDE from observed graph)
    ax6 = Axis(fig[3, 1]; xlabel=L"\mathrm{opinion}", ylabel=L"\mathrm{density}",
               title=L"\mathrm{6.\ Estimated\ opinion\ distribution}", ax_kw...)
    lines!(ax6, collect(xs), y_kde, linewidth=2, color=:firebrick)
    # overlay the latent f(x) as faint reference
    lines!(ax6, collect(xs), y_f, linewidth=1.5, color=(:steelblue, 0.3), linestyle=:dash)

    # Colorbar below the graph, inside a sub-layout of panel 5's column
    Colorbar(fig[4, 3];
        colormap=cgrad([:blue, :white, :red]),
        limits=(-4.0, 4.0),
        label=L"\mathrm{opinion}",
        height=10,
        vertical=false,
        ticklabelsize=10,
        labelsize=12)

    # --- Transformation labels: arrow and description in separate sub-rows ---
    # Top row: arrow on top (large, bold), description below
    let gl = GridLayout(fig[1, 2])
        Label(gl[1, 1]; text="\u2192", fontsize=40, font=:bold, halign=:center, valign=:bottom)
        Label(gl[2, 1]; text="\u00d7 a(x)", fontsize=16, halign=:center, valign=:top)
        rowsize!(gl, 1, Relative(0.6))
    end
    let gl = GridLayout(fig[1, 4])
        Label(gl[1, 1]; text="\u2192", fontsize=40, font=:bold, halign=:center, valign=:bottom)
        Label(gl[2, 1]; text="\u00d7 t(x)", fontsize=16, halign=:center, valign=:top)
        rowsize!(gl, 1, Relative(0.6))
    end
    # Bottom row: description on top, arrow below (large, bold)
    let gl = GridLayout(fig[3, 4])
        Label(gl[1, 1]; text="form edges", fontsize=16, halign=:center, valign=:bottom)
        Label(gl[2, 1]; text="\u2190", fontsize=40, font=:bold, halign=:center, valign=:top)
        rowsize!(gl, 2, Relative(0.6))
    end
    let gl = GridLayout(fig[3, 2])
        Label(gl[1, 1]; text="estimate f", fontsize=16, halign=:center, valign=:bottom)
        Label(gl[2, 1]; text="\u2190", fontsize=40, font=:bold, halign=:center, valign=:top)
        rowsize!(gl, 2, Relative(0.6))
    end
    # Vertical arrow: description on top, arrow below (large, bold)
    let gl = GridLayout(fig[2, 5])
        Label(gl[1, 1]; text="\u00d7 p(x)", fontsize=16, halign=:center, valign=:center)
        Label(gl[2, 1]; text="\u2193", fontsize=40, font=:bold, halign=:center, valign=:center)
        rowsize!(gl, 2, Relative(0.6))
    end

    # --- Sizing ---
    for col in [1, 3, 5]
        colsize!(fig.layout, col, Relative(0.27))
    end
    for col in [2, 4]
        colsize!(fig.layout, col, Relative(0.08))
    end
    rowsize!(fig.layout, 1, Fixed(225))
    rowsize!(fig.layout, 2, Fixed(70))
    rowsize!(fig.layout, 3, Fixed(225))
    rowsize!(fig.layout, 4, Fixed(30))
    rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "fig1_pipeline_anatomy.pdf"), fig)
    save(joinpath(outdir, "fig1_pipeline_anatomy.png"), fig, px_per_unit=2)
    println("Saved fig1_pipeline_anatomy (PDF + PNG)")
end

figure_pipeline_anatomy()
