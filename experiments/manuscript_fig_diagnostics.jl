using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using AlgebraOfGraphics, CairoMakie, LaTeXStrings
using DataFrames, CSV, Statistics

function figure5_diagnostics(; tau_main::Float64=1.0)
    df = CSV.read(joinpath(@__DIR__, "..", "output", "experiments", "parameter_sweep.csv"), DataFrame)
    df_valid = dropmissing(df, :Q)

    df_corr = CSV.read(joinpath(@__DIR__, "..", "output", "experiments", "degree_opinion_corr.csv"), DataFrame)

    # --- Main figure: one tau slice ---
    sub = filter(r -> r.tau == tau_main, df_valid)

    agg = combine(groupby(sub, [:beta, :alpha]),
        :hartigan_dip => (x -> mean(skipmissing(x))) => :mean_dip,
        :valley_peak_ratio => (x -> mean(skipmissing(x))) => :mean_vpr,
    )

    corr_sub = filter(r -> r.tau == tau_main, df_corr)
    agg_corr = combine(groupby(corr_sub, [:beta, :alpha]),
        :degree_opinion_corr => mean => :mean_doc,
    )

    agg = leftjoin(agg, agg_corr, on=[:beta, :alpha])

    set_aog_theme!()
    fig = Figure(size=(1000, 300))

    beta_vals = sort(unique(agg.beta))
    colors = Makie.wong_colors()

    # Column 1: Hartigan dip
    ax1 = Axis(fig[1, 1], xlabel=L"\alpha", ylabel=L"\mathrm{Hartigan\ dip}",
               title=L"\mathrm{Bimodality\ (dip\ test)}")
    for (i, bv) in enumerate(beta_vals)
        s = sort(filter(r -> r.beta == bv, agg), :alpha)
        lines!(ax1, s.alpha, s.mean_dip, linewidth=2, color=colors[i])
    end

    # Column 2: Valley/peak ratio
    ax2 = Axis(fig[1, 2], xlabel=L"\alpha", ylabel=L"\mathrm{valley/peak\ ratio}",
               title=L"\mathrm{Valley\ depth}")
    for (i, bv) in enumerate(beta_vals)
        s = sort(filter(r -> r.beta == bv, agg), :alpha)
        lines!(ax2, s.alpha, s.mean_vpr, linewidth=2, color=colors[i])
    end

    # Column 3: Degree-opinion correlation
    ax3 = Axis(fig[1, 3], xlabel=L"\alpha",
               ylabel=L"\mathrm{Corr}(k_i, |x_i|)",
               title=L"\mathrm{Degree\text{-}opinion\ correlation}")
    for (i, bv) in enumerate(beta_vals)
        s = sort(filter(r -> r.beta == bv, agg), :alpha)
        if !any(ismissing, s.mean_doc)
            lines!(ax3, s.alpha, s.mean_doc, linewidth=2, color=colors[i])
        end
    end

    # Shared legend
    Legend(fig[1, 4],
           [LineElement(color=colors[i], linewidth=2) for i in eachindex(beta_vals)],
           [latexstring("\\beta = " * string(bv)) for bv in beta_vals],
           L"\mathrm{Homophily}",
           framevisible=false)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "fig5_diagnostics.pdf"), fig)
    save(joinpath(outdir, "fig5_diagnostics.png"), fig, px_per_unit=3)
    println("Saved fig5_diagnostics (tau=" * string(tau_main) * ")")
end

function figure5_appendix_all_tau()
    df = CSV.read(joinpath(@__DIR__, "..", "output", "experiments", "parameter_sweep.csv"), DataFrame)
    df_valid = dropmissing(df, :Q)
    df_corr = CSV.read(joinpath(@__DIR__, "..", "output", "experiments", "degree_opinion_corr.csv"), DataFrame)

    tau_vals = sort(unique(df_valid.tau))
    beta_vals = sort(unique(df_valid.beta))
    colors = Makie.wong_colors()

    set_aog_theme!()
    fig = Figure(size=(1000, 250 * length(tau_vals)))

    for (t_idx, tau_val) in enumerate(tau_vals)
        sub = filter(r -> r.tau == tau_val, df_valid)
        agg = combine(groupby(sub, [:beta, :alpha]),
            :hartigan_dip => (x -> mean(skipmissing(x))) => :mean_dip,
            :valley_peak_ratio => (x -> mean(skipmissing(x))) => :mean_vpr,
        )
        corr_sub = filter(r -> r.tau == tau_val, df_corr)
        agg_corr = combine(groupby(corr_sub, [:beta, :alpha]),
            :degree_opinion_corr => mean => :mean_doc,
        )
        agg = leftjoin(agg, agg_corr, on=[:beta, :alpha])

        ax1 = Axis(fig[t_idx, 1], xlabel=L"\alpha", ylabel=L"\mathrm{Hartigan\ dip}",
                    title=latexstring("\\tau = " * string(tau_val)))
        ax2 = Axis(fig[t_idx, 2], xlabel=L"\alpha", ylabel=L"\mathrm{valley/peak}")
        ax3 = Axis(fig[t_idx, 3], xlabel=L"\alpha",
                    ylabel=L"\mathrm{Corr}(k_i, |x_i|)")

        for (i, bv) in enumerate(beta_vals)
            s = sort(filter(r -> r.beta == bv, agg), :alpha)
            lines!(ax1, s.alpha, s.mean_dip, linewidth=2, color=colors[i])
            lines!(ax2, s.alpha, s.mean_vpr, linewidth=2, color=colors[i])
            if !any(ismissing, s.mean_doc)
                lines!(ax3, s.alpha, s.mean_doc, linewidth=2, color=colors[i])
            end
        end
    end

    Legend(fig[:, 4],
           [LineElement(color=colors[i], linewidth=2) for i in eachindex(beta_vals)],
           [latexstring("\\beta = " * string(bv)) for bv in beta_vals],
           L"\mathrm{Homophily}",
           framevisible=false)

    outdir = joinpath(@__DIR__, "..", "output", "manuscript")
    mkpath(outdir)
    save(joinpath(outdir, "figA_diagnostics_all_tau.pdf"), fig)
    save(joinpath(outdir, "figA_diagnostics_all_tau.png"), fig, px_per_unit=3)
    println("Saved figA_diagnostics_all_tau (appendix)")
end

# --- Run ---
figure5_diagnostics(tau_main=1.0)
figure5_appendix_all_tau()
