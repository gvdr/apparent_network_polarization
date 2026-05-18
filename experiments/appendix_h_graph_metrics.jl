# Graph-level metric sweep for Appendix H.
# Two non-homophilic generators:
#   (1) single-topic activity kernel K(x,y) = a(x) * a(y) with
#       a(x) = 1 + tanh(alpha * x), sweeping alpha;
#   (2) mirrored Gaussian opportunity-structure kernel, sweeping mu at
#       fixed sigma.
# Reports per-parameter mean, standard deviation, and monotonicity
# fraction (share of adjacent parameter steps on which the replicate
# mean moves in the expected direction).

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Graphs
using Distributions
using CairoMakie
using Statistics
using Random

println("=== Appendix H: graph metrics under non-homophilic generators ===")
println()

function generate_single_topic_graph(rng, opinions, alpha, target_density)
    N = length(opinions)
    g = SimpleGraph(N)
    a_vals = [1.0 + tanh(alpha * x) for x in opinions]
    raw_sum = 0.0
    for i in 1:N, j in (i+1):N
        raw_sum += a_vals[i] * a_vals[j]
    end
    scale = target_density * N * (N - 1) / (2.0 * raw_sum)
    for i in 1:N, j in (i+1):N
        p = clamp(scale * a_vals[i] * a_vals[j], 0.0, 1.0)
        if rand(rng) < p
            add_edge!(g, i, j)
        end
    end
    return g
end

function generate_mirrored_gaussian_graph(rng, opinions, mu, sigma, target_density)
    N = length(opinions)
    g = SimpleGraph(N)
    lR = [exp(-(x - mu)^2 / (2.0 * sigma^2)) for x in opinions]
    lL = [exp(-(x + mu)^2 / (2.0 * sigma^2)) for x in opinions]
    raw_sum = 0.0
    for i in 1:N, j in (i+1):N
        raw_sum += lR[i] * lR[j] + lL[i] * lL[j]
    end
    scale = target_density * N * (N - 1) / (2.0 * raw_sum)
    for i in 1:N, j in (i+1):N
        kij = lR[i] * lR[j] + lL[i] * lL[j]
        p = clamp(scale * kij, 0.0, 1.0)
        if rand(rng) < p
            add_edge!(g, i, j)
        end
    end
    return g
end

function metrics_for(graph, opinions)
    part_oracle = oracle_sign_partition(opinions)
    bp = part_oracle
    q  = modularity(graph, part_oracle)
    r_raw = try
                rwc(graph, bp)
            catch
                NaN
            end
    r  = (r_raw === missing) ? NaN : r_raw
    ei = ei_index(graph, bp)
    return q, r, ei
end

function summarize_sweep(rng, opinions_sampler, generator, param_values, n_replicates)
    q_mu  = Float64[];  q_sd  = Float64[]
    r_mu  = Float64[];  r_sd  = Float64[]
    ei_mu = Float64[];  ei_sd = Float64[]
    for p in param_values
        qs = Float64[]; rs = Float64[]; es = Float64[]
        for _ in 1:n_replicates
            opinions = opinions_sampler(rng)
            g = generator(rng, opinions, p)
            if ne(g) == 0
                continue
            end
            q, r, e = metrics_for(g, opinions)
            push!(qs, q)
            if !isnan(r)
                push!(rs, r)
            end
            push!(es, e)
        end
        push!(q_mu,  isempty(qs) ? NaN : mean(qs));  push!(q_sd,  isempty(qs) ? NaN : std(qs))
        push!(r_mu,  isempty(rs) ? NaN : mean(rs));  push!(r_sd,  isempty(rs) ? NaN : std(rs))
        push!(ei_mu, isempty(es) ? NaN : mean(es));  push!(ei_sd, isempty(es) ? NaN : std(es))
    end
    return q_mu, q_sd, r_mu, r_sd, ei_mu, ei_sd
end

function monotone_fraction(xs; direction = :up)
    d = diff(xs)
    d = d[.!isnan.(d)]
    if isempty(d)
        return NaN
    end
    return direction == :up ? sum(d .>= 0.0) / length(d) : sum(d .<= 0.0) / length(d)
end

N              = 600
target_density = 0.03
n_replicates   = 6
rng            = Xoshiro(11)
opinions_gaussian(r) = randn(r, N)

alpha_values = collect(range(0.0, 2.5, length=10))
q_mu_s, q_sd_s, r_mu_s, r_sd_s, ei_mu_s, ei_sd_s =
    summarize_sweep(rng, opinions_gaussian,
                    (r, x, p) -> generate_single_topic_graph(r, x, p, target_density),
                    alpha_values, n_replicates)
for (i, alpha) in enumerate(alpha_values)
    println("single-topic alpha = " * string(round(alpha; digits=3)) *
            ": Q = "   * string(round(q_mu_s[i];  digits=3)) * " +/- " * string(round(q_sd_s[i];  digits=3)) *
            ", RWC = " * string(round(r_mu_s[i];  digits=3)) * " +/- " * string(round(r_sd_s[i];  digits=3)) *
            ", EI = "  * string(round(ei_mu_s[i]; digits=3)) * " +/- " * string(round(ei_sd_s[i]; digits=3)))
end

mu_values   = collect(range(0.0, 2.5, length=10))
sigma_fixed = 1.0
q_mu_g, q_sd_g, r_mu_g, r_sd_g, ei_mu_g, ei_sd_g =
    summarize_sweep(rng, opinions_gaussian,
                    (r, x, p) -> generate_mirrored_gaussian_graph(r, x, p, sigma_fixed, target_density),
                    mu_values, n_replicates)
for (i, mu) in enumerate(mu_values)
    println("mirrored-gauss mu = " * string(round(mu; digits=3)) *
            ": Q = "   * string(round(q_mu_g[i];  digits=3)) * " +/- " * string(round(q_sd_g[i];  digits=3)) *
            ", RWC = " * string(round(r_mu_g[i];  digits=3)) * " +/- " * string(round(r_sd_g[i];  digits=3)) *
            ", EI = "  * string(round(ei_mu_g[i]; digits=3)) * " +/- " * string(round(ei_sd_g[i]; digits=3)))
end

println()
println("Monotonicity fraction (share of adjacent parameter steps in the expected direction):")
println("  single-topic:    Q up = "  * string(round(monotone_fraction(q_mu_s; direction=:up);   digits=2)) *
                         ",  RWC up = " * string(round(monotone_fraction(r_mu_s; direction=:up);   digits=2)) *
                         ",  EI down = " * string(round(monotone_fraction(ei_mu_s; direction=:down); digits=2)))
println("  mirrored Gauss:  Q up = "  * string(round(monotone_fraction(q_mu_g; direction=:up);   digits=2)) *
                         ",  RWC up = " * string(round(monotone_fraction(r_mu_g; direction=:up);   digits=2)) *
                         ",  EI down = " * string(round(monotone_fraction(ei_mu_g; direction=:down); digits=2)))

mkpath(joinpath(@__DIR__, "..", "output", "experiments"))
fig = Figure(size=(960, 540))
ax1 = Axis(fig[1, 1], xlabel="activity asymmetry alpha", ylabel="modularity (Q)",
           title="Single-topic generator")
ax2 = Axis(fig[1, 2], xlabel="activity asymmetry alpha", ylabel="RWC")
ax3 = Axis(fig[1, 3], xlabel="activity asymmetry alpha", ylabel="EI")
ax4 = Axis(fig[2, 1], xlabel="topic separation mu", ylabel="modularity (Q)",
           title="Mirrored Gaussian generator")
ax5 = Axis(fig[2, 2], xlabel="topic separation mu", ylabel="RWC")
ax6 = Axis(fig[2, 3], xlabel="topic separation mu", ylabel="EI")

for (ax, mus, sds) in [(ax1, q_mu_s, q_sd_s), (ax2, r_mu_s, r_sd_s), (ax3, ei_mu_s, ei_sd_s)]
    lo = mus .- sds
    hi = mus .+ sds
    band!(ax, alpha_values, lo, hi; color = (:royalblue, 0.2))
    lines!(ax, alpha_values, mus; color = :royalblue)
    scatter!(ax, alpha_values, mus; color = :royalblue)
end

for (ax, mus, sds) in [(ax4, q_mu_g, q_sd_g), (ax5, r_mu_g, r_sd_g), (ax6, ei_mu_g, ei_sd_g)]
    lo = mus .- sds
    hi = mus .+ sds
    band!(ax, mu_values, lo, hi; color = (:darkorange, 0.2))
    lines!(ax, mu_values, mus; color = :darkorange)
    scatter!(ax, mu_values, mus; color = :darkorange)
end

save(joinpath(@__DIR__, "..", "output", "experiments", "appendix_h_graph_metrics.png"), fig)

println()
println("Figures saved under output/experiments/.")
