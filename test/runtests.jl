using Test
using PolarizationPipeline
using Distributions
using Graphs
using Random
using Statistics

@testset "PolarizationPipeline" begin
    @testset "PipelineConfig" begin
        @testset "default construction" begin
            config = PipelineConfig()
            @test config.N == 500
            @test config.opinion_dist isa Normal
            @test config.ρ == 0.05
            @test config.activity(0.0) == 1.0
            @test config.activity(1.0) == 1.0
            @test config.homophily(0.0, 0.0) == 1.0
            @test config.topic_filter(0.0) == 1.0
        end

        @testset "manuscript_config_A" begin
            config = manuscript_config_A(α=2.0, β=1.0, τ=1.0)
            @test config.activity(0.0) == 1.0
            @test config.activity(1.0) == 3.0
            @test config.homophily(0.0, 1.0) < 1.0
            @test config.topic_filter(0.0) == 0.5
            # topic filter clamped to 1.0 for large |x|
            @test config.topic_filter(10.0) <= 1.0
        end

        @testset "manuscript_config_P" begin
            config = manuscript_config_P(μ=2.0, τ_comp=0.5)
            @test config.opinion_dist isa MixtureModel
            @test config.activity(5.0) == 1.0
        end

        @testset "linear_homophily_config" begin
            config = linear_homophily_config(L=3.0, κ=0.1)
            @test config.opinion_dist isa Uniform
            @test config.homophily(0.0, 0.0) == 1.0
            @test config.homophily(0.0, 1.0) ≈ 0.9
            @test config.activity(5.0) == 1.0
        end
    end

    @testset "opinions" begin
        config = PipelineConfig(N=1000, rng=Xoshiro(42))
        opinions = sample_opinions(config)
        @test length(opinions) == 1000
        @test eltype(opinions) == Float64
        # mean should be near 0 for N(0,1)
        @test abs(mean(opinions)) < 0.2

        # reproducible with same seed
        config2 = PipelineConfig(N=1000, rng=Xoshiro(42))
        opinions2 = sample_opinions(config2)
        @test opinions == opinions2
    end

    @testset "networks" begin
        rng = Xoshiro(42)
        opinions = [0.0, 0.0, 0.0, 3.0, 3.0]
        config = PipelineConfig(
            N=5, ρ=1.0,
            activity = x -> 1.0,
            homophily = (x, y) -> exp(-abs(x - y)),
            rng=rng
        )
        g = form_network(opinions, config)
        @test g isa SimpleGraph
        @test nv(g) == 5
        @test ne(g) > 0

        # extreme nodes should have higher degree with activity bias
        config_large = PipelineConfig(
            N=200, ρ=0.1,
            activity = x -> 1.0 + 2.0 * x^2,
            homophily = (x, y) -> exp(-abs(x - y)),
            opinion_dist = Normal(0.0, 1.0),
            rng=Xoshiro(42)
        )
        ops = sample_opinions(config_large)
        g_large = form_network(ops, config_large)
        extreme_mask = abs.(ops) .> 1.0
        moderate_mask = abs.(ops) .< 0.5
        if sum(extreme_mask) > 5 && sum(moderate_mask) > 5
            extreme_deg = mean(degree(g_large, findall(extreme_mask)))
            moderate_deg = mean(degree(g_large, findall(moderate_mask)))
            @test extreme_deg > moderate_deg
        end
    end

    @testset "sampling" begin
        @testset "apply_topic_filter" begin
            opinions = [-2.0, -1.0, 0.0, 1.0, 2.0]
            config = PipelineConfig(
                topic_filter = x -> min(1.0, 0.5 + 0.25 * x^2),
                rng = Xoshiro(42)
            )
            mask = apply_topic_filter(opinions, config)
            @test mask isa BitVector
            @test length(mask) == 5
        end

        @testset "extract_giant_component" begin
            g = SimpleGraph(5)
            add_edge!(g, 1, 2)
            add_edge!(g, 2, 3)
            # nodes 4,5 isolated
            mask = BitVector([true, true, true, true, false])
            sub_g, indices = extract_giant_component(g, mask)
            @test nv(sub_g) == 3
            @test sort(indices) == [1, 2, 3]
        end

        @testset "stage2_visibility" begin
            g = SimpleGraph(4)
            add_edge!(g, 1, 2)
            add_edge!(g, 1, 3)
            add_edge!(g, 1, 4)
            vis = stage2_visibility(g)
            @test length(vis) == 4
            @test vis[1] > vis[2]
            @test mean(vis) ≈ 1.0
        end

        @testset "sample_observed" begin
            opinions = randn(Xoshiro(42), 100)
            config = PipelineConfig(
                N=100, ρ=0.3,
                activity = x -> 1.0,
                homophily = (x, y) -> exp(-0.5 * abs(x - y)),
                topic_filter = x -> 1.0,
                rng = Xoshiro(42)
            )
            g = form_network(opinions, config)
            obs_g, obs_ops, obs_idx = sample_observed(g, opinions, config)
            @test nv(obs_g) <= 100
            @test length(obs_ops) == nv(obs_g)
            @test length(obs_idx) == nv(obs_g)
            @test obs_ops == opinions[obs_idx]
        end

        @testset "activity-biased observation favors off-center nodes" begin
            config = PipelineConfig(
                N=400, ρ=0.08,
                activity = x -> 1.0 + 2.0 * x^2,
                homophily = (x, y) -> exp(-0.5 * abs(x - y)),
                topic_filter = x -> 1.0,
                opinion_dist = Normal(0.0, 1.0),
                rng = Xoshiro(7)
            )
            opinions = sample_opinions(config)
            g = form_network(opinions, config)
            obs_g, obs_ops, obs_idx = sample_observed(g, opinions, config)
            @test nv(obs_g) > 0
            @test mean(abs.(obs_ops)) > mean(abs.(opinions))
            @test mean(degree(g, obs_idx)) >= mean(degree(g))
        end
    end

    @testset "partitioning" begin
        @testset "oracle_sign_partition" begin
            opinions = [-2.0, -1.0, 0.5, 1.0, 2.0]
            part = oracle_sign_partition(opinions)
            @test part[1] == part[2]  # both negative
            @test part[3] == part[4] == part[5]  # all positive
            @test part[1] != part[3]  # different sides
            @test Set(part) == Set([1, 2])
        end

        @testset "community_partition" begin
            g = SimpleGraph(20)
            for i in 1:10, j in (i+1):10
                add_edge!(g, i, j)
            end
            for i in 11:20, j in (i+1):20
                add_edge!(g, i, j)
            end
            add_edge!(g, 5, 15)
            part = community_partition(g)
            @test length(part) == 20
            @test length(unique(part)) >= 2
        end

        @testset "binarize_partition" begin
            # already binary: should be a relabel
            g = SimpleGraph(4)
            add_edge!(g, 1, 2); add_edge!(g, 3, 4)
            part = [1, 1, 2, 2]
            bp = binarize_partition(g, part)
            @test Set(bp) == Set([1, 2])
            @test bp[1] == bp[2]
            @test bp[3] == bp[4]

            # three communities: merge smallest into nearest
            g3 = SimpleGraph(6)
            add_edge!(g3, 1, 2)  # comm A
            add_edge!(g3, 3, 4)  # comm B
            add_edge!(g3, 5, 6)  # comm C (small, tied)
            add_edge!(g3, 5, 3)  # C connects to B
            part3 = [1, 1, 2, 2, 3, 3]
            bp3 = binarize_partition(g3, part3)
            @test Set(bp3) == Set([1, 2])
            @test bp3[5] == bp3[3]  # C merged with B
        end
    end

    @testset "densities" begin
        @testset "find_modes" begin
            f = x -> pdf(Normal(0, 1), x)
            modes = find_modes(f; grid_range=(-5.0, 5.0))
            @test length(modes) == 1
            @test abs(modes[1]) < 0.1

            f_bi = x -> 0.5 * pdf(Normal(-2, 0.5), x) + 0.5 * pdf(Normal(2, 0.5), x)
            modes_bi = find_modes(f_bi; grid_range=(-5.0, 5.0))
            @test length(modes_bi) == 2
        end

        @testset "valley_to_peak_ratio" begin
            f_bi = x -> 0.5 * pdf(Normal(-2, 0.5), x) + 0.5 * pdf(Normal(2, 0.5), x)
            modes = find_modes(f_bi; grid_range=(-5.0, 5.0))
            vtp = valley_to_peak_ratio(f_bi, modes)
            @test vtp < 1.0
            @test vtp >= 0.0
        end
    end

    @testset "run_pipeline" begin
        config = manuscript_config_A(α=1.0, β=1.0, τ=0.0, N=100, ρ=0.1, rng=Xoshiro(42))
        result = run_pipeline(config)
        @test result isa PipelineResult
        @test length(result.opinions) == 100
        @test nv(result.obs_graph) <= 100
        @test length(result.obs_opinions) == nv(result.obs_graph)
        @test length(result.obs_indices) == nv(result.obs_graph)
        @test result.obs_opinions == result.opinions[result.obs_indices]
        @test length(result.partition) == nv(result.obs_graph)
        @test length(result.binary_partition) == nv(result.obs_graph)
        @test Set(result.binary_partition) ⊆ Set([1, 2])
        @test haskey(result.metrics, :Q)
    end

    @testset "diagnostics" begin
        @testset "bimodality_coefficient" begin
            rng = Xoshiro(42)
            unimodal = randn(rng, 10000)
            bc_uni = bimodality_coefficient(unimodal)
            @test bc_uni < 0.555

            bimodal = vcat(randn(rng, 5000) .- 3.0, randn(rng, 5000) .+ 3.0)
            bc_bi = bimodality_coefficient(bimodal)
            @test bc_bi > 0.555
        end

        @testset "hartigan_dip" begin
            rng = Xoshiro(123)
            unimodal = randn(rng, 4000)
            bimodal = vcat(randn(rng, 2000) .- 2.5, randn(rng, 2000) .+ 2.5)
            dip_uni = hartigan_dip(unimodal)
            dip_bi = hartigan_dip(bimodal)
            @test dip_uni >= 0.0
            @test dip_bi > dip_uni
            @test dip_bi > 0.01
        end

        @testset "sample_valley_peak_ratio" begin
            rng = Xoshiro(321)
            unimodal = randn(rng, 4000)
            bimodal = vcat(randn(rng, 2000) .- 2.5, randn(rng, 2000) .+ 2.5)
            vpr_uni = sample_valley_peak_ratio(unimodal)
            vpr_bi = sample_valley_peak_ratio(bimodal)
            @test 0.0 <= vpr_uni <= 1.0
            @test 0.0 <= vpr_bi <= 1.0
            @test vpr_bi < vpr_uni
        end

        @testset "degree_opinion_correlation" begin
            g = SimpleGraph(10)
            for i in 1:5, j in (i+1):5
                add_edge!(g, i, j)
            end
            for i in 6:10, j in (i+1):10
                add_edge!(g, i, j)
            end
            opinions = Float64[1,2,3,4,5, 6,7,8,9,10]
            part = [1,1,1,1,1, 2,2,2,2,2]
            corrs = degree_opinion_correlation(g, opinions, part)
            @test length(corrs) == 2
            @test all(c -> c isa Float64, corrs)
        end
    end

    @testset "normalization" begin
        @testset "configuration_model_sample" begin
            g = SimpleGraph(20)
            for i in 1:10, j in (i+1):10
                add_edge!(g, i, j)
            end
            for i in 11:20, j in (i+1):20
                add_edge!(g, i, j)
            end
            rng = Xoshiro(42)
            g_cm = configuration_model_sample(g, rng)
            @test nv(g_cm) == nv(g)
            @test ne(g_cm) == ne(g)
        end

        @testset "normalized_scores" begin
            g = SimpleGraph(20)
            for i in 1:10, j in (i+1):10
                add_edge!(g, i, j)
            end
            for i in 11:20, j in (i+1):20
                add_edge!(g, i, j)
            end
            add_edge!(g, 5, 15)
            rng = Xoshiro(42)
            ns = normalized_scores(g, community_partition; n_randomizations=10, rng=rng)
            @test haskey(ns, :Q_norm)
            @test haskey(ns, :rwc_norm)
            @test ns.Q_norm isa Float64
        end
    end

    @testset "metrics" begin
        # two clear cliques with one bridge
        g = SimpleGraph(10)
        for i in 1:5, j in (i+1):5
            add_edge!(g, i, j)
        end
        for i in 6:10, j in (i+1):10
            add_edge!(g, i, j)
        end
        add_edge!(g, 5, 6)
        part_full = [1,1,1,1,1, 2,2,2,2,2]
        part_binary = [1,1,1,1,1, 2,2,2,2,2]

        @testset "modularity" begin
            Q = PolarizationPipeline.modularity(g, part_full)
            @test Q isa Float64
            @test Q > 0.0
            Q_one = PolarizationPipeline.modularity(g, ones(Int, 10))
            @test abs(Q_one) < 1e-10
        end

        @testset "ei_index" begin
            ei = ei_index(g, part_binary)
            @test ei isa Float64
            @test -1.0 <= ei <= 1.0
            @test ei < 0.0  # more internal than external
        end

        @testset "spectral_gap" begin
            sg = spectral_gap(g)
            @test sg isa Float64
            @test sg > 0.0
            @test sg < 2.0
        end

        @testset "rwc" begin
            r = rwc(g, part_binary; k=2)
            if !ismissing(r)
                @test r isa Float64
            end
        end

        @testset "compute_all_metrics" begin
            m = compute_all_metrics(g, part_full, part_binary; k=2)
            @test haskey(m, :Q)
            @test haskey(m, :rwc)
            @test haskey(m, :spectral_gap)
            @test haskey(m, :ei)
        end
    end

    @testset "edge cases" begin
        @testset "empty graph from topic filter" begin
            config = PipelineConfig(
                N=50, ρ=0.1,
                topic_filter = x -> 0.0,  # reject all nodes
                rng = Xoshiro(42)
            )
            result = run_pipeline(config)
            @test result isa PipelineResult
            @test nv(result.obs_graph) == 0
            @test isempty(result.obs_indices)
            @test isempty(result.partition)
            @test ismissing(result.metrics.rwc)
        end

        @testset "negative homophily clamped" begin
            config = PipelineConfig(
                N=20, ρ=1.0,
                homophily = (x, y) -> 1.0 - 10.0 * abs(x - y),  # can go very negative
                opinion_dist = Uniform(-3.0, 3.0),
                rng = Xoshiro(42)
            )
            opinions = sample_opinions(config)
            g = form_network(opinions, config)
            @test g isa SimpleGraph  # no error thrown
            @test nv(g) == 20
        end

        @testset "degree_opinion_correlation zero variance" begin
            # clique: all degrees identical within community
            g = SimpleGraph(6)
            for i in 1:3, j in (i+1):3
                add_edge!(g, i, j)
            end
            for i in 4:6, j in (i+1):6
                add_edge!(g, i, j)
            end
            opinions = Float64[1, 2, 3, 4, 5, 6]
            part = [1, 1, 1, 2, 2, 2]
            corrs = degree_opinion_correlation(g, opinions, part)
            @test length(corrs) == 2
            @test all(c -> !isnan(c), corrs)  # no NaN
            @test all(c -> c == 0.0, corrs)   # zero variance in degree -> 0.0
        end
    end
end
