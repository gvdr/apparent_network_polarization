using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using DataFrames, CSV, Statistics

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------

datadir  = joinpath(@__DIR__, "..", "output", "experiments")
outdir   = joinpath(@__DIR__, "..", "output", "manuscript")
mkpath(outdir)

sweep_path = joinpath(datadir, "parameter_sweep.csv")
if !isfile(sweep_path)
    error("parameter_sweep.csv not found. Run parameter_sweep.jl first.")
end

df_raw = CSV.read(sweep_path, DataFrame)
println("Loaded " * string(nrow(df_raw)) * " rows from parameter_sweep.csv")

# Categorical beta labels (used as color)
df_raw.beta_label = string.(df_raw.beta)

# Categorical tau labels (used as row facet)
# Build proper LaTeXStrings for each tau value
df_raw.tau_label = [latexstring("\\tau = " * string(t)) for t in df_raw.tau]

# Aggregate across replications
agg = combine(
    groupby(df_raw, [:beta, :alpha, :tau, :beta_label, :tau_label]),
    :Q               => mean => :mean_Q,
    :Q_norm           => mean => :mean_Q_norm,
    :ei               => mean => :mean_ei,
    :bimodality_coeff => mean => :mean_bc,
    :hartigan_dip     => mean => :mean_dip,
    :valley_peak_ratio => mean => :mean_vpr,
)
sort!(agg, [:tau, :beta, :alpha])

set_aog_theme!()

# ---------------------------------------------------------------------------
# Figure 5 -- Bimodality diagnostics (3 columns x 3 tau rows)
# ---------------------------------------------------------------------------

function figure5(agg, outdir)
    # Pivot to long form with a "diagnostic" column
    rows = NamedTuple[]
    for r in eachrow(agg)
        base = (alpha = r.alpha, beta_label = r.beta_label, tau_label = r.tau_label)
        push!(rows, (; base..., diagnostic = "Hartigan dip",     value = r.mean_dip))
        push!(rows, (; base..., diagnostic = "Valley/peak ratio", value = r.mean_vpr))
        push!(rows, (; base..., diagnostic = "Sarle BC",          value = r.mean_bc))
    end
    df_long = DataFrame(rows)

    # Use sorter to control column facet order
    diag_sorter = sorter("Hartigan dip", "Valley/peak ratio", "Sarle BC")

    plt = data(df_long) *
        mapping(
            :alpha  => L"\alpha",
            :value  => "Diagnostic value",
            color   = :beta_label => L"\beta",
            row     = :tau_label,
            col     = :diagnostic => diag_sorter,
        ) *
        visual(Lines, linewidth = 2)

    fg = draw(plt;
              figure = (size = (1100, 700),),
              legend = (position = :right,))

    # Add the Sarle threshold line (0.555) in the rightmost column
    all_axes = [c for c in contents(fg.figure.layout) if c isa Axis]
    for ax in all_axes
        gc = ax.layoutobservables.gridcontent[]
        if gc.span.cols.start == 3  # third column = Sarle BC
            hlines!(ax, [0.555]; color = :red, linestyle = :dash, linewidth = 1)
        end
    end

    save(joinpath(outdir, "fig5_bimodality_diagnostics.pdf"), fg)
    save(joinpath(outdir, "fig5_bimodality_diagnostics.png"), fg, px_per_unit = 3)
    println("Saved fig5_bimodality_diagnostics")
end

# ---------------------------------------------------------------------------
# Figure 6 -- Raw vs normalized modularity (2 columns x 3 tau rows)
# ---------------------------------------------------------------------------

function figure6(agg, outdir)
    rows = NamedTuple[]
    for r in eachrow(agg)
        base = (alpha = r.alpha, beta_label = r.beta_label, tau_label = r.tau_label)
        push!(rows, (; base..., metric = L"Q\ \mathrm{(raw)}",       value = r.mean_Q))
        push!(rows, (; base..., metric = L"Q\ \mathrm{(z\text{-}score)}", value = r.mean_Q_norm))
    end
    df_long = DataFrame(rows)

    plt = data(df_long) *
        mapping(
            :alpha  => L"\alpha",
            :value  => "Modularity",
            color   = :beta_label => L"\beta",
            row     = :tau_label,
            col     = :metric,
        ) *
        visual(Lines, linewidth = 2)

    fg = draw(plt;
              figure = (size = (800, 700),),
              legend = (position = :right,))

    save(joinpath(outdir, "fig6_raw_vs_normalized_Q.pdf"), fg)
    save(joinpath(outdir, "fig6_raw_vs_normalized_Q.png"), fg, px_per_unit = 3)
    println("Saved fig6_raw_vs_normalized_Q")
end

# ---------------------------------------------------------------------------
# Figure 4 -- EI-index vs alpha (1 column x 3 tau rows)
# ---------------------------------------------------------------------------

function figure4(agg, outdir)
    plt = data(agg) *
        mapping(
            :alpha    => L"\alpha",
            :mean_ei  => L"\mathrm{EI}",
            color     = :beta_label => L"\beta",
            row       = :tau_label,
        ) *
        visual(Lines, linewidth = 2)

    fg = draw(plt;
              figure = (size = (600, 700),),
              legend = (position = :right,))

    save(joinpath(outdir, "fig4_ei_vs_alpha.pdf"), fg)
    save(joinpath(outdir, "fig4_ei_vs_alpha.png"), fg, px_per_unit = 3)
    println("Saved fig4_ei_vs_alpha")
end

# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------

figure5(agg, outdir)
figure6(agg, outdir)
figure4(agg, outdir)

println("All manuscript experiment figures saved to " * outdir)
