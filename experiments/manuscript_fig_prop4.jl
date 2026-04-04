using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using Distributions, DataFrames, Statistics, StatsBase, Random

function figure_prop4()
    sigma = 1.0
    alpha = 2.0
    rng = Xoshiro(42)
    f = Normal(0.0, sigma)
    phi = x -> 1.0 + alpha * x^2
    max_phi = 1.0 + alpha * 16.0  # phi at 4sigma

    n_values = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    n_mc = 20  # replications per n

    rows = NamedTuple[]
    for n in n_values
        for mc in 1:n_mc
            # sample from f_obs via rejection sampling
            samples = Float64[]
            while length(samples) < n
                x = rand(rng, f)
                if rand(rng) < phi(x) / max_phi
                    push!(samples, x)
                end
            end
            # add consistent-estimator noise (shrinks with n)
            noise_std = 0.5 / sqrt(Float64(n))
            estimates = samples .+ randn(rng, n) .* noise_std

            ecdf_est = ecdf(estimates)

            # large reference sample for F_obs
            ref = Float64[]
            while length(ref) < 50_000
                x = rand(rng, f)
                if rand(rng) < phi(x) / max_phi
                    push!(ref, x)
                end
            end
            ecdf_fobs = ecdf(ref)

            test_pts = range(-4.0, 4.0, length=500)
            ks_fobs = maximum(abs(ecdf_est(t) - ecdf_fobs(t)) for t in test_pts)
            ks_f = maximum(abs(ecdf_est(t) - cdf(f, t)) for t in test_pts)

            push!(rows, (n=n, ks=ks_fobs, target=L"F_{\mathrm{obs}}"))
            push!(rows, (n=n, ks=ks_f, target=L"F"))
        end
    end

    df = DataFrame(rows)

    # aggregate: mean (and std) per (n, target)
    agg = combine(
        groupby(df, [:n, :target]),
        :ks => mean => :mean_ks,
        :ks => std  => :std_ks,
    )

    set_aog_theme!()

    plt = data(agg) *
        mapping(
            :n => (x -> log10(Float64(x))) => L"\log_{10}(n)",
            :mean_ks => L"\mathrm{KS\ distance}",
            color = :target => "Convergence target",
        ) *
        visual(ScatterLines, linewidth=2, markersize=8)

    fig = draw(
        plt,
        axis = (
            title = L"\mathrm{Prop\ 4:\ scores\ converge\ to}\ F_{\mathrm{obs}}\mathrm{,\ not}\ F",
            xticks = (
                log10.([50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]),
                ["50", "100", "200", "500", "1000", "2000", "5000", "10000"],
            ),
        ),
        figure = (size = (600, 400),),
    )

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "fig7_prop4_convergence.pdf"), fig)
    save(joinpath(outdir, "fig7_prop4_convergence.png"), fig)
    println("Saved fig7_prop4_convergence to " * outdir)
end

figure_prop4()
