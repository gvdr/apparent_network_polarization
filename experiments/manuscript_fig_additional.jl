using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using DataFrames, CSV, Statistics, StatsBase
using PolarizationPipeline, Graphs, GraphMakie
using GraphMakie.NetworkLayout: Spring
using Random, Distributions

outdir = joinpath(@__DIR__, "..", "output", "manuscript")
mkpath(outdir)

# ---------------------------------------------------------------------------
# Figure: RWC vs alpha (from experiment data, tau=1.0 only)
# ---------------------------------------------------------------------------

function figure_rwc(outdir)
    set_aog_theme!()

    datapath = joinpath(@__DIR__, "..", "output", "experiments", "parameter_sweep.csv")
    if !isfile(datapath)
        error("parameter_sweep.csv not found. Run parameter_sweep.jl first.")
    end
    df = CSV.read(datapath, DataFrame)
    println("Loaded " * string(nrow(df)) * " rows from parameter_sweep.csv")

    # Filter to tau=1.0 and drop missing RWC
    sub = dropmissing(filter(r -> r.tau == 1.0, df), :rwc)
    println("After filtering tau=1.0 and dropping missing RWC: " * string(nrow(sub)) * " rows")

    if nrow(sub) == 0
        println("WARNING: no valid RWC rows for tau=1.0, skipping figure_rwc")
        return
    end

    agg = combine(
        groupby(sub, [:beta, :alpha]),
        :rwc => mean => :mean_rwc,
    )
    sort!(agg, [:beta, :alpha])
    agg.beta_label = string.(agg.beta)

    plt = data(agg) *
        mapping(
            :alpha    => L"\alpha",
            :mean_rwc => L"\mathrm{RWC}",
            color     = :beta_label => L"\beta",
        ) *
        visual(Lines, linewidth = 2)

    fig = draw(plt;
               figure = (size = (600, 400),),
               legend = (position = :right,))

    save(joinpath(outdir, "fig_rwc_vs_alpha.pdf"), fig)
    save(joinpath(outdir, "fig_rwc_vs_alpha.png"), fig, px_per_unit = 3)
    println("Saved fig_rwc_vs_alpha")
end

# ---------------------------------------------------------------------------
# Figure: Prop 3bis oracle separation (AoG style)
# ---------------------------------------------------------------------------

function figure_prop3bis_separation(outdir)
    set_aog_theme!()

    n_samples = 200_000
    rng = Xoshiro(42)
    beta_values = [0.5, 1.0, 2.0, 5.0]
    alpha_values = range(0.0, 5.0, length = 20)

    rows = NamedTuple[]
    for beta in beta_values
        for alpha in alpha_values
            x = randn(rng, n_samples)
            weights = [1.0 + alpha * xi^2 for xi in x]
            weights ./= sum(weights)

            u = abs.(x)
            # Resample according to weights to get weighted E[min(U,V)]
            idx1 = wsample(rng, 1:n_samples, weights, n_samples)
            u_samp = u[idx1]
            idx2 = wsample(rng, 1:n_samples, weights, n_samples)
            v_samp = u[idx2]
            E_min = mean(min.(u_samp, v_samp))
            S = 2.0 * beta * E_min

            push!(rows, (alpha = alpha, beta = beta, S = S,
                         beta_label = string(beta)))
        end
    end

    df = DataFrame(rows)

    plt = data(df) *
        mapping(
            :alpha      => L"\alpha",
            :S          => L"S(g) = 2\beta\,\mathbb{E}[\min(U,V)]",
            color       = :beta_label => L"\beta",
        ) *
        visual(Lines, linewidth = 2)

    fig = draw(plt;
               figure = (size = (600, 400),),
               legend = (position = :right,))

    save(joinpath(outdir, "fig_prop3bis_separation.pdf"), fig)
    save(joinpath(outdir, "fig_prop3bis_separation.png"), fig, px_per_unit = 3)
    println("Saved fig_prop3bis_separation")
end

# ---------------------------------------------------------------------------
# Figure: Model A vs Model P network comparison
# ---------------------------------------------------------------------------

function figure_model_comparison(outdir)
    set_aog_theme!()

    N = 500
    beta = 1.5
    rho = 0.04
    target_edges = 400  # fixed observation budget (like sampling N tweets)

    # -- Model A: full pipeline with activity bias --
    result_A = run_pipeline(manuscript_config_A(
        α=2.0, β=beta, τ=1.0, N=N, ρ=rho, rng=Xoshiro(123)))

    # -- Model P: full pipeline, genuinely bimodal, flat activity --
    result_P = run_pipeline(manuscript_config_P(
        μ=1.5, τ_comp=0.5, β=beta, N=N, ρ=rho, rng=Xoshiro(456)))

    # -- Subsample edges to a fixed budget (simulating data collection) --
    function subsample_graph(graph, opinions, n_edges, rng)
        all_edges = collect(edges(graph))
        if length(all_edges) <= n_edges
            return graph, opinions
        end
        sampled = all_edges[randperm(rng, length(all_edges))[1:n_edges]]
        # find nodes involved in sampled edges
        node_set = Set{Int}()
        for e in sampled
            push!(node_set, src(e))
            push!(node_set, dst(e))
        end
        node_list = sort(collect(node_set))
        node_map = Dict(v => i for (i, v) in enumerate(node_list))
        g_new = SimpleGraph(length(node_list))
        for e in sampled
            add_edge!(g_new, node_map[src(e)], node_map[dst(e)])
        end
        ops_new = opinions[node_list]
        # giant component
        cc = connected_components(g_new)
        gc_idx = cc[argmax(length.(cc))]
        g_gc, _ = induced_subgraph(g_new, gc_idx)
        ops_gc = ops_new[gc_idx]
        return g_gc, ops_gc
    end

    gc_A, ops_A = subsample_graph(result_A.obs_graph, result_A.obs_opinions, target_edges, Xoshiro(789))
    gc_P, ops_P = subsample_graph(result_P.obs_graph, result_P.obs_opinions, target_edges, Xoshiro(789))

    println("Model A observed: " * string(nv(gc_A)) * " nodes, " * string(ne(gc_A)) * " edges")
    println("Model P observed: " * string(nv(gc_P)) * " nodes, " * string(ne(gc_P)) * " edges")

    # -- Color mapping: use cgrad, symmetric around 0 --
    cmap = cgrad([:blue, :white, :red])
    op_lim = max(maximum(abs.(ops_A)), maximum(abs.(ops_P)), 3.0)
    function opinion_colors(opinions)
        return [cmap[clamp((o + op_lim) / (2.0 * op_lim), 0.0, 1.0)] for o in opinions]
    end

    colors_A = opinion_colors(ops_A)
    colors_P = opinion_colors(ops_P)

    fig = Figure(size=(1000, 500))

    # Left panel: Model A
    ax_A = Axis(fig[1, 1],
        title=L"\mathrm{Observed\ graph\ (Model\ A:\ unimodal\ +\ activity\ bias)}")
    hidespines!(ax_A)
    hidedecorations!(ax_A)

    graphplot!(ax_A, gc_A;
        layout=Spring(; seed=1),
        node_color=colors_A,
        node_size=8,
        edge_width=0.5,
        edge_color=(:gray50, 0.25))

    # Right panel: Model P
    ax_P = Axis(fig[1, 2],
        title=L"\mathrm{Observed\ graph\ (Model\ P:\ bimodal\ +\ flat\ activity)}")
    hidespines!(ax_P)
    hidedecorations!(ax_P)

    graphplot!(ax_P, gc_P;
        layout=Spring(; seed=1),
        node_color=colors_P,
        node_size=8,
        edge_width=0.5,
        edge_color=(:gray50, 0.25))

    # Shared colorbar
    Colorbar(fig[2, 1:2],
        colormap=cmap,
        limits=(-op_lim, op_lim),
        label=L"\mathrm{opinion}",
        vertical=false,
        flipaxis=false,
        width=Relative(0.4))

    save(joinpath(outdir, "fig_model_comparison.pdf"), fig)
    save(joinpath(outdir, "fig_model_comparison.png"), fig, px_per_unit=3)
    println("Saved fig_model_comparison")
end

# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

println("=== Generating additional manuscript figures ===")
figure_rwc(outdir)
figure_prop3bis_separation(outdir)
figure_model_comparison(outdir)
println("All additional figures saved to " * outdir)
