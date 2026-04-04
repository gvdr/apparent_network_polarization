using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Graphs
using DataFrames
using CSV
using Statistics
using Random
using ProgressMeter

function run_calibration()
println("=== Model P Calibration ===")

# --- Parameters ---
betas = [0.5, 1.0, 2.0]
alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
taus = [0.0, 1.0, 3.0]
N = 1000
rho = 0.05
n_reps_target = 10
n_reps_candidate = 5
max_attempts_per_rep = 3

# Model P search grid
mu_coarse = range(0.5, 4.0, length=8)
tau_comp_coarse = range(0.3, 1.5, length=6)

println("Coarse grid: " * string(length(mu_coarse)) * " x " * string(length(tau_comp_coarse)) * " = " * string(length(mu_coarse) * length(tau_comp_coarse)) * " candidates per triple")

# --- Helper: collect scores from a config, handling degeneracy ---
function collect_scores(config_fn, n_reps; max_attempts_per_rep::Int=3)
    Qs = Float64[]
    eis = Float64[]
    sgs = Float64[]
    rwcs = Float64[]
    n_degenerate_attempts = 0
    for rep in 1:n_reps
        found_valid = false
        for attempt in 0:(max_attempts_per_rep - 1)
            result = run_pipeline(config_fn(rep, attempt))
            if nv(result.obs_graph) < 4
                n_degenerate_attempts += 1
                continue
            end
            push!(Qs, result.metrics.Q)
            push!(eis, result.metrics.ei)
            push!(sgs, result.metrics.spectral_gap)
            if !ismissing(result.metrics.rwc)
                push!(rwcs, result.metrics.rwc)
            end
            found_valid = true
            break
        end
        if !found_valid
            continue
        end
    end
    if isempty(Qs)
        return nothing
    end
    return (Q=mean(Qs), ei=mean(eis), spectral_gap=mean(sgs),
            rwc=isempty(rwcs) ? missing : mean(rwcs),
            n_valid=length(Qs),
            complete=(length(Qs) == n_reps),
            n_degenerate_attempts=n_degenerate_attempts)
end

# --- Compute Model A target scores + their std devs for standardization (threaded) ---
println("Computing Model A target scores...")
target_triples = [(beta, alpha, tau) for beta in betas for alpha in alphas for tau in taus]
target_results = Vector{Union{Nothing, Tuple{Tuple{Float64,Float64,Float64}, NamedTuple, NamedTuple}}}(nothing, length(target_triples))

@showprogress Threads.@threads for idx in eachindex(target_triples)
    beta, alpha, tau = target_triples[idx]
    config_fn = (rep, attempt) -> begin
        seed = hash((beta, alpha, tau, rep, attempt, :target))
        manuscript_config_A(α=alpha, β=beta, τ=tau, N=N, ρ=rho, rng=Xoshiro(seed))
    end
    target = collect_scores(config_fn, n_reps_target; max_attempts_per_rep=max_attempts_per_rep)
    if !isnothing(target) && target.complete
        Qs = Float64[]; eis = Float64[]; sgs = Float64[]; rwcs = Float64[]
        for rep in 1:n_reps_target
            for attempt in 0:(max_attempts_per_rep - 1)
                result = run_pipeline(config_fn(rep, attempt))
                if nv(result.obs_graph) < 4
                    continue
                end
                push!(Qs, result.metrics.Q)
                push!(eis, result.metrics.ei)
                push!(sgs, result.metrics.spectral_gap)
                if !ismissing(result.metrics.rwc)
                    push!(rwcs, result.metrics.rwc)
                end
                break
            end
        end
        scores = (
            Q=mean(Qs), ei=mean(eis), spectral_gap=mean(sgs),
            rwc=isempty(rwcs) ? missing : mean(rwcs),
            n_valid=length(Qs))
        stds = (
            Q=max(std(Qs), 1e-8),
            ei=max(std(eis), 1e-8),
            spectral_gap=max(std(sgs), 1e-8),
            rwc=length(rwcs) > 1 ? max(std(rwcs), 1e-8) : 1.0)
        target_results[idx] = ((beta, alpha, tau), scores, stds)
    end
end

# collect into dicts
target_scores = Dict{Tuple{Float64,Float64,Float64}, NamedTuple}()
target_stds = Dict{Tuple{Float64,Float64,Float64}, NamedTuple}()
for tr in target_results
    if !isnothing(tr)
        key, scores, stds = tr
        target_scores[key] = scores
        target_stds[key] = stds
    end
end

# --- Standardized distance ---
function score_distance(target, candidate, sd)
    d = ((target.Q - candidate.Q) / sd.Q)^2 +
        ((target.ei - candidate.ei) / sd.ei)^2 +
        ((target.spectral_gap - candidate.spectral_gap) / sd.spectral_gap)^2
    # include RWC if both sides have it
    if !ismissing(target.rwc) && !ismissing(candidate.rwc)
        d += ((target.rwc - candidate.rwc) / sd.rwc)^2
    end
    return sqrt(d)
end

# --- Grid search (threaded) ---
println("Searching for Model P matches...")

# filter to triples that have target scores
search_triples = [(beta, alpha, tau) for (beta, alpha, tau) in target_triples
                  if haskey(target_scores, (beta, alpha, tau))]
rows_grid = Vector{Union{Nothing, NamedTuple}}(nothing, length(search_triples))

outdir = joinpath(@__DIR__, "..", "output", "experiments")
mkpath(outdir)

try
@showprogress Threads.@threads for tidx in eachindex(search_triples)
    beta, alpha, tau = search_triples[tidx]
    key = (beta, alpha, tau)
    target = target_scores[key]
    sd = target_stds[key]

    best_mu = 1.0
    best_tau_comp = 0.5
    best_dist = Inf

    # coarse grid — sequential within this triple (each triple runs on one thread)
    for mu in mu_coarse, tc in tau_comp_coarse
        config_fn = (rep, attempt) -> manuscript_config_P(
            μ=mu, τ_comp=tc, β=beta, N=N, ρ=rho,
            rng=Xoshiro(hash((beta, mu, tc, rep, attempt, :calibration))))
        candidate = collect_scores(config_fn, n_reps_candidate; max_attempts_per_rep=max_attempts_per_rep)
        if isnothing(candidate) || !candidate.complete
            continue
        end
        d = score_distance(target, candidate, sd)
        if d < best_dist
            best_dist = d
            best_mu = mu
            best_tau_comp = tc
        end
    end

    # fine grid around best coarse
    mu_step = (mu_coarse[2] - mu_coarse[1]) / 2.0
    tc_step = (tau_comp_coarse[2] - tau_comp_coarse[1]) / 2.0
    mu_fine = range(max(0.1, best_mu - mu_step), best_mu + mu_step, length=7)
    tc_fine = range(max(0.1, best_tau_comp - tc_step), best_tau_comp + tc_step, length=7)

    for mu in mu_fine, tc in tc_fine
        config_fn = (rep, attempt) -> manuscript_config_P(
            μ=mu, τ_comp=tc, β=beta, N=N, ρ=rho,
            rng=Xoshiro(hash((beta, mu, tc, rep, attempt, :calibration_fine))))
        candidate = collect_scores(config_fn, n_reps_candidate; max_attempts_per_rep=max_attempts_per_rep)
        if isnothing(candidate) || !candidate.complete
            continue
        end
        d = score_distance(target, candidate, sd)
        if d < best_dist
            best_dist = d
            best_mu = mu
            best_tau_comp = tc
        end
    end

    rows_grid[tidx] = (
        beta=beta, alpha=alpha, tau=tau,
        target_Q=target.Q, target_ei=target.ei,
        target_spectral_gap=target.spectral_gap,
        target_rwc=target.rwc,
        target_n_valid=target.n_valid,
        mu_equiv=best_mu, tau_comp_equiv=best_tau_comp,
        match_distance=best_dist,
    )

    println("beta=" * string(beta) * " alpha=" * string(alpha) * " tau=" * string(tau) *
            " -> mu=" * string(round(best_mu, digits=3)) * " tc=" * string(round(best_tau_comp, digits=3)) *
            " dist=" * string(round(best_dist, digits=4)))
end

finally
rows = collect(filter(!isnothing, rows_grid))
df = DataFrame(rows)
CSV.write(joinpath(outdir, "model_p_calibration.csv"), df)
println("Saved " * string(nrow(df)) * " rows to model_p_calibration.csv")
end # try/finally
end

run_calibration()
