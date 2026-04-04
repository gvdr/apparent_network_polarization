using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using DataFrames
using CSV
using CairoMakie
using Statistics

println("=== Plotting Results ===")

datadir = joinpath(@__DIR__, "..", "output", "experiments")
figdir = joinpath(datadir, "figures")
mkpath(figdir)

# --- Load data ---
sweep_path = joinpath(datadir, "parameter_sweep.csv")
if !isfile(sweep_path)
    println("No parameter_sweep.csv found. Run parameter_sweep.jl first.")
    exit(1)
end

df = CSV.read(sweep_path, DataFrame)
# drop degenerate rows for plotting
df_valid = dropmissing(df, :Q)
println("Loaded " * string(nrow(df)) * " rows (" * string(nrow(df_valid)) * " valid)")

# --- Compute inflation factors ---
baselines = combine(
    groupby(filter(r -> r.alpha == 0.0 && r.tau == 0.0, df_valid), :beta),
    :Q => mean => :Q_base,
    :ei => mean => :ei_base,
    :spectral_gap => mean => :sg_base,
)

df_inf = leftjoin(df_valid, baselines, on=:beta)
df_inf.Q_inflation = df_inf.Q ./ df_inf.Q_base

# --- Figure 1: Heatmaps of Q inflation across (alpha, tau) for each beta ---
for beta_val in sort(unique(df_valid.beta))
    sub = filter(r -> r.beta == beta_val, df_inf)
    agg = combine(groupby(sub, [:alpha, :tau]), :Q_inflation => mean => :mean_Q_inflation)
    alphas = sort(unique(agg.alpha))
    taus_vals = sort(unique(agg.tau))
    matrix = zeros(length(alphas), length(taus_vals))
    for row in eachrow(agg)
        i = findfirst(==(row.alpha), alphas)
        j = findfirst(==(row.tau), taus_vals)
        matrix[i, j] = row.mean_Q_inflation
    end
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], xlabel="alpha", ylabel="tau",
              title="Q inflation (beta=" * string(beta_val) * ")")
    hm = heatmap!(ax, alphas, taus_vals, matrix)
    Colorbar(fig[1, 2], hm, label="Q / Q_baseline")
    save(joinpath(figdir, "Q_inflation_beta_" * string(beta_val) * ".png"), fig)
end
println("Q inflation heatmaps saved.")

# --- Figure 2: Raw vs normalized Q (all tau values) ---
fig2 = Figure(size=(900, 600))
for (t_idx, tau_val) in enumerate(sort(unique(df_valid.tau)))
    ax1 = Axis(fig2[t_idx, 1], xlabel="alpha", ylabel="raw Q",
               title="Raw Q (tau=" * string(tau_val) * ")")
    ax2 = Axis(fig2[t_idx, 2], xlabel="alpha", ylabel="Q z-score",
               title="Normalized Q (tau=" * string(tau_val) * ")")
    for beta_val in sort(unique(df_valid.beta))
        sub = filter(r -> r.beta == beta_val && r.tau == tau_val, df_valid)
        agg = combine(groupby(sub, :alpha), :Q => mean => :mean_Q, :Q_norm => mean => :mean_Q_norm)
        sort!(agg, :alpha)
        lines!(ax1, agg.alpha, agg.mean_Q, label="beta=" * string(beta_val))
        lines!(ax2, agg.alpha, agg.mean_Q_norm, label="beta=" * string(beta_val))
    end
    if t_idx == 1
        axislegend(ax1)
        axislegend(ax2)
    end
end
save(joinpath(figdir, "raw_vs_normalized_Q.png"), fig2)
println("Raw vs normalized Q saved.")

# --- Figure 3: RWC and normalized RWC (all tau values) ---
df_rwc = dropmissing(df_valid, :rwc)
if nrow(df_rwc) > 0
    fig3 = Figure(size=(900, 600))
    for (t_idx, tau_val) in enumerate(sort(unique(df_rwc.tau)))
        ax1 = Axis(fig3[t_idx, 1], xlabel="alpha", ylabel="raw RWC",
                    title="Raw RWC (tau=" * string(tau_val) * ")")
        ax2 = Axis(fig3[t_idx, 2], xlabel="alpha", ylabel="RWC z-score",
                    title="Normalized RWC (tau=" * string(tau_val) * ")")
        df_rwc_norm = dropmissing(filter(r -> r.tau == tau_val, df_valid), [:rwc, :rwc_norm])
        for beta_val in sort(unique(df_rwc.beta))
            sub = filter(r -> r.beta == beta_val && r.tau == tau_val, df_rwc)
            if nrow(sub) > 0
                agg = combine(groupby(sub, :alpha), :rwc => mean => :mean_rwc)
                sort!(agg, :alpha)
                lines!(ax1, agg.alpha, agg.mean_rwc, label="beta=" * string(beta_val))
            end
            sub_n = filter(r -> r.beta == beta_val, df_rwc_norm)
            if nrow(sub_n) > 0
                agg_n = combine(groupby(sub_n, :alpha), :rwc_norm => mean => :mean_rwc_norm)
                sort!(agg_n, :alpha)
                lines!(ax2, agg_n.alpha, agg_n.mean_rwc_norm, label="beta=" * string(beta_val))
            end
        end
        if t_idx == 1
            axislegend(ax1)
            axislegend(ax2)
        end
    end
    save(joinpath(figdir, "raw_vs_normalized_RWC.png"), fig3)
    println("RWC figures saved.")
end

# --- Figure 4: Bimodality diagnostics (all tau) ---
fig4 = Figure(size=(1400, length(unique(df_valid.tau)) * 300))
for (t_idx, tau_val) in enumerate(sort(unique(df_valid.tau)))
    ax_bc = Axis(fig4[t_idx, 1], xlabel="alpha", ylabel="bimodality coefficient",
                 title="Sarle coefficient (tau=" * string(tau_val) * ")")
    ax_dip = Axis(fig4[t_idx, 2], xlabel="alpha", ylabel="Hartigan dip",
                  title="Hartigan dip (tau=" * string(tau_val) * ")")
    ax_vpr = Axis(fig4[t_idx, 3], xlabel="alpha", ylabel="valley/peak ratio",
                  title="Valley/peak ratio (tau=" * string(tau_val) * ")")
    for beta_val in sort(unique(df_valid.beta))
        sub = filter(r -> r.beta == beta_val && r.tau == tau_val, df_valid)
        agg = combine(groupby(sub, :alpha),
                      :bimodality_coeff => mean => :mean_bc,
                      :hartigan_dip => mean => :mean_dip,
                      :valley_peak_ratio => mean => :mean_vpr)
        sort!(agg, :alpha)
        lines!(ax_bc, agg.alpha, agg.mean_bc, label="beta=" * string(beta_val))
        lines!(ax_dip, agg.alpha, agg.mean_dip, label="beta=" * string(beta_val))
        lines!(ax_vpr, agg.alpha, agg.mean_vpr, label="beta=" * string(beta_val))
    end
    hlines!(ax_bc, [0.555], color=:red, linestyle=:dash, label="Sarle threshold")
    if t_idx == 1
        axislegend(ax_bc)
        axislegend(ax_dip)
        axislegend(ax_vpr)
    end
end
save(joinpath(figdir, "bimodality_diagnostics_vs_alpha.png"), fig4)
println("Bimodality diagnostics saved.")

# --- Figure 5: EI-index (all tau) ---
fig5 = Figure(size=(900, length(unique(df_valid.tau)) * 300))
for (t_idx, tau_val) in enumerate(sort(unique(df_valid.tau)))
    ax = Axis(fig5[t_idx, 1], xlabel="alpha", ylabel="EI-index",
              title="EI-index (tau=" * string(tau_val) * ")")
    for beta_val in sort(unique(df_valid.beta))
        sub = filter(r -> r.beta == beta_val && r.tau == tau_val, df_valid)
        agg = combine(groupby(sub, :alpha), :ei => mean => :mean_ei)
        sort!(agg, :alpha)
        lines!(ax, agg.alpha, agg.mean_ei, label="beta=" * string(beta_val))
    end
    if t_idx == 1
        axislegend(ax)
    end
end
save(joinpath(figdir, "ei_vs_alpha.png"), fig5)
println("EI figures saved.")

# --- Figure 6: Spectral gap (all tau) ---
fig6 = Figure(size=(900, length(unique(df_valid.tau)) * 300))
for (t_idx, tau_val) in enumerate(sort(unique(df_valid.tau)))
    ax = Axis(fig6[t_idx, 1], xlabel="alpha", ylabel="spectral gap",
              title="Spectral gap (tau=" * string(tau_val) * ")")
    for beta_val in sort(unique(df_valid.beta))
        sub = filter(r -> r.beta == beta_val && r.tau == tau_val, df_valid)
        agg = combine(groupby(sub, :alpha), :spectral_gap => mean => :mean_sg)
        sort!(agg, :alpha)
        lines!(ax, agg.alpha, agg.mean_sg, label="beta=" * string(beta_val))
    end
    if t_idx == 1
        axislegend(ax)
    end
end
save(joinpath(figdir, "spectral_gap_vs_alpha.png"), fig6)
println("Spectral gap figures saved.")

# --- Figure 7: Calibration (if available) ---
calib_path = joinpath(datadir, "model_p_calibration.csv")
if isfile(calib_path)
    df_cal = CSV.read(calib_path, DataFrame)
    tau_vals = sort(unique(df_cal.tau))
    fig7 = Figure(size=(900, length(tau_vals) * 300))
    for (t_idx, tau_val) in enumerate(tau_vals)
        ax1 = Axis(fig7[t_idx, 1], xlabel="alpha",
                   ylabel="equivalent mu",
                   title="Equivalent mode separation (tau=" * string(tau_val) * ")")
        ax2 = Axis(fig7[t_idx, 2], xlabel="alpha",
                   ylabel="equivalent tau_comp",
                   title="Equivalent component spread (tau=" * string(tau_val) * ")")
        for beta_val in sort(unique(df_cal.beta))
            sub = filter(r -> r.beta == beta_val && r.tau == tau_val, df_cal)
            sort!(sub, :alpha)
            lines!(ax1, sub.alpha, sub.mu_equiv, label="beta=" * string(beta_val))
            lines!(ax2, sub.alpha, sub.tau_comp_equiv, label="beta=" * string(beta_val))
        end
        if t_idx == 1
            axislegend(ax1)
            axislegend(ax2)
        end
    end
    save(joinpath(figdir, "model_p_equivalence.png"), fig7)
    println("Calibration figures saved.")
end

println("All figures saved to " * figdir)
