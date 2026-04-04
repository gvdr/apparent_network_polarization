using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using DataFrames, CSV, Statistics

function figure6_raw_vs_normalized(; tau_main::Float64=1.0)
    df = CSV.read(joinpath(@__DIR__, "..", "output", "experiments", "parameter_sweep.csv"), DataFrame)
    df_valid = dropmissing(df, :Q)

    sub = filter(r -> r.tau == tau_main, df_valid)
    agg = combine(groupby(sub, [:beta, :alpha]),
        :Q => mean => :mean_Q,
        :Q_norm => mean => :mean_Q_norm,
    )

    beta_vals = sort(unique(agg.beta))
    colors = Makie.wong_colors()

    set_aog_theme!()
    fig = Figure(size=(700, 300))

    ax1 = Axis(fig[1, 1], xlabel=L"\alpha", ylabel=L"Q",
               title=L"\mathrm{Raw\ modularity}")
    for (i, bv) in enumerate(beta_vals)
        s = sort(filter(r -> r.beta == bv, agg), :alpha)
        lines!(ax1, s.alpha, s.mean_Q, linewidth=2, color=colors[i])
    end

    ax2 = Axis(fig[1, 2], xlabel=L"\alpha", ylabel=L"Q\ z\mathrm{\text{-}score}",
               title=L"\mathrm{Normalized\ modularity}")
    for (i, bv) in enumerate(beta_vals)
        s = sort(filter(r -> r.beta == bv, agg), :alpha)
        lines!(ax2, s.alpha, s.mean_Q_norm, linewidth=2, color=colors[i])
    end

    Legend(fig[1, 3],
           [LineElement(color=colors[i], linewidth=2) for i in eachindex(beta_vals)],
           [latexstring("\\beta = " * string(bv)) for bv in beta_vals],
           L"\mathrm{Homophily}",
           framevisible=false)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "fig6_raw_vs_normalized_Q.pdf"), fig)
    save(joinpath(outdir, "fig6_raw_vs_normalized_Q.png"), fig, px_per_unit=3)
    println("Saved fig6_raw_vs_normalized_Q (tau=" * string(tau_main) * ")")
end

function figure6_appendix_all_tau()
    df = CSV.read(joinpath(@__DIR__, "..", "output", "experiments", "parameter_sweep.csv"), DataFrame)
    df_valid = dropmissing(df, :Q)

    tau_vals = sort(unique(df_valid.tau))
    beta_vals = sort(unique(df_valid.beta))
    colors = Makie.wong_colors()

    set_aog_theme!()
    fig = Figure(size=(700, 250 * length(tau_vals)))

    for (t_idx, tau_val) in enumerate(tau_vals)
        sub = filter(r -> r.tau == tau_val, df_valid)
        agg = combine(groupby(sub, [:beta, :alpha]),
            :Q => mean => :mean_Q,
            :Q_norm => mean => :mean_Q_norm,
        )

        ax1 = Axis(fig[t_idx, 1], xlabel=L"\alpha", ylabel=L"Q",
                    title=latexstring("\\mathrm{Raw}\\ Q,\\ \\tau = " * string(tau_val)))
        ax2 = Axis(fig[t_idx, 2], xlabel=L"\alpha", ylabel=L"Q\ z\mathrm{\text{-}score}",
                    title=latexstring("\\mathrm{Normalized}\\ Q,\\ \\tau = " * string(tau_val)))

        for (i, bv) in enumerate(beta_vals)
            s = sort(filter(r -> r.beta == bv, agg), :alpha)
            lines!(ax1, s.alpha, s.mean_Q, linewidth=2, color=colors[i])
            lines!(ax2, s.alpha, s.mean_Q_norm, linewidth=2, color=colors[i])
        end
    end

    Legend(fig[:, 3],
           [LineElement(color=colors[i], linewidth=2) for i in eachindex(beta_vals)],
           [latexstring("\\beta = " * string(bv)) for bv in beta_vals],
           L"\mathrm{Homophily}",
           framevisible=false)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "figB_raw_vs_normalized_Q_all_tau.pdf"), fig)
    save(joinpath(outdir, "figB_raw_vs_normalized_Q_all_tau.png"), fig, px_per_unit=3)
    println("Saved figB_raw_vs_normalized_Q_all_tau (appendix)")
end

# --- Run ---
figure6_raw_vs_normalized(tau_main=1.0)
figure6_appendix_all_tau()
