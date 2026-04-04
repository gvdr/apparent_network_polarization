function sample_opinions(config::PipelineConfig)
    return rand(config.rng, config.opinion_dist, config.N)
end
