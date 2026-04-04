function apply_topic_filter(opinions::Vector{Float64}, config::PipelineConfig)
    rng = config.rng
    t = config.topic_filter
    mask = BitVector(undef, length(opinions))
    for i in eachindex(opinions)
        p = clamp(t(opinions[i]), 0.0, 1.0)
        mask[i] = rand(rng) < p
    end
    return mask
end

function stage2_visibility(graph::SimpleGraph)
    deg = degree(graph)
    mean_deg = mean(deg)
    if mean_deg <= 0
        return zeros(Float64, nv(graph))
    end
    return deg ./ mean_deg
end

function apply_observation_filter(graph::SimpleGraph, opinions::Vector{Float64}, config::PipelineConfig)
    rng = config.rng
    t = config.topic_filter
    vis = stage2_visibility(graph)
    mask = BitVector(undef, length(opinions))
    for i in eachindex(opinions)
        p = clamp(t(opinions[i]) * vis[i], 0.0, 1.0)
        mask[i] = rand(rng) < p
    end
    return mask
end

function extract_giant_component(graph::SimpleGraph, mask::BitVector)
    retained = findall(mask)
    if isempty(retained)
        return SimpleGraph(0), Int[]
    end
    sub_g, vmap = induced_subgraph(graph, retained)
    cc = connected_components(sub_g)
    largest_cc = cc[argmax(length.(cc))]
    gc_g, gc_vmap = induced_subgraph(sub_g, largest_cc)
    original_indices = retained[largest_cc]
    return gc_g, original_indices
end

function sample_observed(graph::SimpleGraph, opinions::Vector{Float64}, config::PipelineConfig)
    mask = apply_observation_filter(graph, opinions, config)
    obs_graph, obs_indices = extract_giant_component(graph, mask)
    obs_opinions = opinions[obs_indices]
    return obs_graph, obs_opinions, obs_indices
end
