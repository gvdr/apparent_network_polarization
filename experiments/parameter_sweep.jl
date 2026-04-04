using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Graphs
using DataFrames
using CSV
using Statistics
using Random
using ProgressMeter

function run_parameter_sweep()
    println("=== Parameter Sweep (Section 4) ===")

    # --- Parameters ---
    betas = [0.5, 1.0, 2.0]
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    taus = [0.0, 1.0, 3.0]
    n_reps = 30
    N = 1000
    rho = 0.05
    n_rand = 100   # manuscript specifies 100 configuration model randomizations
    max_attempts = 3  # resample up to this many times per cell to get a valid run

    n_cells = length(betas) * length(alphas) * length(taus)
    println("Parameter cells: " * string(n_cells))
    println("Replications per cell: " * string(n_reps))
    println("Total target runs: " * string(n_cells * n_reps))

    # --- Main sweep (threaded) ---
    all_runs = [(beta, alpha, tau, rep)
                for beta in betas for alpha in alphas for tau in taus for rep in 1:n_reps]
    results_main = Vector{Union{Nothing, NamedTuple}}(nothing, length(all_runs))
    # per-run corr rows — avoids shared mutable state (threadid() unreliable in Julia 1.9+)
    results_corr_per_run = Vector{Vector{NamedTuple}}(undef, length(all_runs))
    for i in eachindex(results_corr_per_run)
        results_corr_per_run[i] = NamedTuple[]
    end
    n_degenerate = Threads.Atomic{Int}(0)

    outdir = joinpath(@__DIR__, "..", "output", "experiments")
    mkpath(outdir)

    try
    @showprogress Threads.@threads for idx in eachindex(all_runs)
        beta, alpha, tau, rep = all_runs[idx]
        seed_base = hash((beta, alpha, tau, rep))
        success = false

        for attempt in 0:(max_attempts - 1)
            seed = seed_base + attempt
            config = manuscript_config_A(
                α=alpha, β=beta, τ=tau, N=N, ρ=rho, rng=Xoshiro(seed)
            )
            result = run_pipeline(config)

            if nv(result.obs_graph) < 4
                Threads.atomic_add!(n_degenerate, 1)
                continue
            end

            # normalized scores
            ns = normalized_scores(result.obs_graph, community_partition;
                                   n_randomizations=n_rand, rng=Xoshiro(seed + 1000))

            bc = bimodality_coefficient(result.obs_opinions)
            dip = hartigan_dip(result.obs_opinions)
            vpr = sample_valley_peak_ratio(result.obs_opinions)

            results_main[idx] = (
                beta=beta, alpha=alpha, tau=tau, seed=seed, rep=rep,
                N_obs=nv(result.obs_graph),
                n_communities=length(unique(result.partition)),
                Q=result.metrics.Q,
                Q_norm=ns.Q_norm,
                rwc=ismissing(result.metrics.rwc) ? missing : result.metrics.rwc,
                rwc_norm=ns.rwc_norm,
                spectral_gap=result.metrics.spectral_gap,
                ei=result.metrics.ei,
                bimodality_coeff=bc,
                hartigan_dip=dip,
                valley_peak_ratio=vpr,
            )

            # per-community diagnostics — stored per-run (each idx is private)
            corrs = degree_opinion_correlation(result.obs_graph, result.obs_opinions, result.partition)
            comm_ids = sort(unique(result.partition))
            for (c_idx, c_id) in enumerate(comm_ids)
                c_size = count(==(c_id), result.partition)
                push!(results_corr_per_run[idx], (
                    beta=beta, alpha=alpha, tau=tau, seed=seed, rep=rep,
                    community_id=c_idx, community_size=c_size,
                    degree_opinion_corr=corrs[c_idx],
                ))
            end

            success = true
            break
        end

        if !success
            results_main[idx] = (
                beta=beta, alpha=alpha, tau=tau, seed=seed_base, rep=rep,
                N_obs=0,
                n_communities=0,
                Q=missing,
                Q_norm=missing,
                rwc=missing,
                rwc_norm=missing,
                spectral_gap=missing,
                ei=missing,
                bimodality_coeff=missing,
                hartigan_dip=missing,
                valley_peak_ratio=missing,
            )
        end
    end

    finally
    # save whatever we have, even on error
    rows_main = filter(!isnothing, results_main)
    rows_corr = reduce(vcat, results_corr_per_run)

    df_main = DataFrame(rows_main)
    CSV.write(joinpath(outdir, "parameter_sweep.csv"), df_main)
    n_valid = count(r -> !ismissing(r.Q), rows_main)
    println("Saved " * string(nrow(df_main)) * " rows (" * string(n_valid) * " valid, " * string(length(rows_main) - n_valid) * " degenerate) to parameter_sweep.csv")
    println("Total degenerate attempts: " * string(n_degenerate[]))

    df_corr = DataFrame(rows_corr)
    CSV.write(joinpath(outdir, "degree_opinion_corr.csv"), df_corr)
    println("Saved " * string(nrow(df_corr)) * " rows to degree_opinion_corr.csv")
    end # try/finally
end

run_parameter_sweep()
