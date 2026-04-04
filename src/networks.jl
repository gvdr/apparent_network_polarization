function form_network(opinions::Vector{Float64}, config::PipelineConfig)
    N = length(opinions)
    g = SimpleGraph(N)
    a = config.activity
    h = config.homophily
    rho = config.ρ
    rng = config.rng
    for i in 1:N
        a_i = a(opinions[i])
        for j in (i+1):N
            p = clamp(rho * a_i * a(opinions[j]) * h(opinions[i], opinions[j]), 0.0, 1.0)
            if rand(rng) < p
                add_edge!(g, i, j)
            end
        end
    end
    return g
end
