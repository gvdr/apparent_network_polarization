using Distributions
using Graphs
using Random

@kwdef struct PipelineConfig{D<:Distribution, F1, F2, F3, R<:AbstractRNG}
    N::Int = 500
    opinion_dist::D = Normal(0.0, 1.0)
    activity::F1 = x -> 1.0
    homophily::F2 = (x, y) -> exp(-abs(x - y))
    topic_filter::F3 = x -> 1.0
    ρ::Float64 = 0.05
    rng::R = Xoshiro()
end

function manuscript_config_A(; α=0.0, β=1.0, τ=0.0, σ=1.0, kwargs...)
    PipelineConfig(;
        opinion_dist = Normal(0.0, σ),
        activity = x -> 1.0 + α * x^2,
        homophily = (x, y) -> exp(-β * abs(x - y)),
        topic_filter = x -> min(1.0, (1.0 + τ * x^2) / (1.0 + τ)),
        kwargs...
    )
end

function manuscript_config_P(; μ=2.0, τ_comp=0.5, β=1.0, kwargs...)
    PipelineConfig(;
        opinion_dist = MixtureModel([Normal(-μ, τ_comp), Normal(μ, τ_comp)]),
        activity = x -> 1.0,
        topic_filter = x -> 1.0,
        homophily = (x, y) -> exp(-β * abs(x - y)),
        kwargs...
    )
end

function linear_homophily_config(; L=3.0, κ=0.1, opinion_dist=Uniform(-L, L), kwargs...)
    PipelineConfig(;
        opinion_dist = opinion_dist,
        homophily = (x, y) -> 1.0 - κ * abs(x - y),
        activity = x -> 1.0,
        topic_filter = x -> 1.0,
        kwargs...
    )
end

const MetricsTuple = @NamedTuple{Q::Float64, rwc::Union{Missing,Float64}, spectral_gap::Float64, ei::Float64}

struct PipelineResult
    opinions::Vector{Float64}
    graph::SimpleGraph
    obs_indices::Vector{Int}
    obs_opinions::Vector{Float64}
    obs_graph::SimpleGraph
    partition::Vector{Int}
    binary_partition::Vector{Int}
    metrics::MetricsTuple
end
