module PolarizationPipeline

using Distributions
using Graphs
using LinearAlgebra
using SparseArrays
using Statistics
using StatsBase
using Random

include("types.jl")
include("opinions.jl")
include("networks.jl")
include("sampling.jl")
include("metrics.jl")
include("normalization.jl")
include("diagnostics.jl")
include("densities.jl")

export PipelineConfig, PipelineResult, MetricsTuple
export manuscript_config_A, manuscript_config_P, linear_homophily_config
export sample_opinions
export form_network
export apply_topic_filter, stage2_visibility, apply_observation_filter, extract_giant_component, sample_observed
export oracle_sign_partition, community_partition, binarize_partition
export ei_index, spectral_gap, rwc, compute_all_metrics
export configuration_model_sample, normalized_scores
export bimodality_coefficient, hartigan_dip, sample_valley_peak_ratio, degree_opinion_correlation
export observed_density, find_modes, valley_to_peak_ratio, heteroskedastic_convolution
export run_pipeline

function run_pipeline(config::PipelineConfig)
    opinions = sample_opinions(config)
    graph = form_network(opinions, config)
    obs_graph, obs_opinions, obs_indices = sample_observed(graph, opinions, config)
    if nv(obs_graph) == 0
        empty_metrics = MetricsTuple((0.0, missing, 0.0, 0.0))
        return PipelineResult(opinions, graph, obs_indices, obs_opinions,
                              obs_graph, Int[], Int[], empty_metrics)
    end
    partition = community_partition(obs_graph)
    binary_partition = binarize_partition(obs_graph, partition)
    metrics = compute_all_metrics(obs_graph, partition, binary_partition)
    return PipelineResult(opinions, graph, obs_indices, obs_opinions,
                          obs_graph, partition, binary_partition, metrics)
end

end # module
