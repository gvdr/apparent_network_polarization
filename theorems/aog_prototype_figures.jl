using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Statistics
using CairoMakie
using AlgebraOfGraphics

const OUTDIR = joinpath(@__DIR__, "..", "output", "theorems")
const STRESSDIR = joinpath(OUTDIR, "stress")

mkpath(OUTDIR)

function prop3_threshold_figure()
    df = CSV.read(joinpath(STRESSDIR, "stress_prop3_boundary.csv"), DataFrame)
    df.signed_distance = df.s .- df.threshold
    df.sigma_label = string.("σ = ", round.(df.sigma; digits=2))
    df.grid_label = string.("grid = ", df.grid_points)
    sort!(df, [:sigma, :side, :grid_points, :signed_distance])

    plt =
        data(df) *
        mapping(
            :signed_distance => "s - threshold",
            :observed_modes => "Observed mode count",
            color = :side => renamer("below" => "Below threshold", "above" => "Above threshold"),
            marker = :grid_label => "Resolution",
            layout = :sigma_label => "Gaussian scale",
        ) *
        visual(Scatter, markersize = 10)

    fig = draw(
        plt,
        axis = (
            xticks = -0.2:0.1:0.2,
            yticks = [1, 2],
            title = "Proposition 3: the bimodality threshold is sharp",
        ),
        figure = (size = (980, 620),),
    )

    save(joinpath(OUTDIR, "prop3_boundary_aog.png"), fig)
    fig
end

function prop4_gap_figure()
    df = CSV.read(joinpath(STRESSDIR, "stress_prop4_replications.csv"), DataFrame)
    df.logn = log10.(df.n)
    summary = combine(groupby(df, :n)) do sdf
        (
            gap_mean = mean(sdf.gap),
            gap_lo = quantile(sdf.gap, 0.1),
            gap_hi = quantile(sdf.gap, 0.9),
        )
    end
    summary.logn = log10.(summary.n)

    points =
        data(df) *
        mapping(
            :logn => "log10(sample size n)",
            :gap => "KS(F) - KS(F_obs)",
        ) *
        visual(Scatter, color = (:steelblue, 0.45), markersize = 9)

    summary_df = DataFrame(
        logn = summary.logn,
        gap_mean = summary.gap_mean,
        gap_lo = summary.gap_lo,
        gap_hi = summary.gap_hi,
    )

    line =
        data(summary_df) *
        mapping(
            :logn => "log10(sample size n)",
            :gap_mean => "KS(F) - KS(F_obs)",
        ) *
        visual(Lines, color = :darkorange, linewidth = 3)

    fig = draw(
        line + points;
        axis = (
            xticks = (log10.([100, 500, 1000, 5000]), ["100", "500", "1000", "5000"]),
            title = "Proposition 4: estimated scores converge toward F_obs, not F",
        ),
        figure = (size = (760, 480),),
    )

    save(joinpath(OUTDIR, "prop4_gap_aog.png"), fig)
    fig
end

function main()
    println("Generating AlgebraOfGraphics prototypes...")
    prop3_threshold_figure()
    prop4_gap_figure()
    println("Saved:")
    println(joinpath(OUTDIR, "prop3_boundary_aog.png"))
    println(joinpath(OUTDIR, "prop4_gap_aog.png"))
end

main()
