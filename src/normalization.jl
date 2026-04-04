function configuration_model_sample(graph::SimpleGraph, rng::AbstractRNG)
    # Use edge-swap Markov chain to preserve exact degree sequence and edge count.
    # Start from a copy of the graph and perform random double-edge swaps.
    g = copy(graph)
    m = ne(g)
    if m < 2
        return g
    end

    n_swaps = 10 * m
    edge_list = collect(edges(g))

    for _ in 1:n_swaps
        # pick two distinct edges at random
        i = rand(rng, 1:length(edge_list))
        j = rand(rng, 1:length(edge_list))
        while j == i
            j = rand(rng, 1:length(edge_list))
        end

        e1 = edge_list[i]
        e2 = edge_list[j]
        u, v = src(e1), dst(e1)
        s, t = src(e2), dst(e2)

        # propose swap: (u,v),(s,t) -> (u,s),(v,t) or (u,t),(v,s)
        # choose one of the two swap variants
        if rand(rng, Bool)
            a, b, c, d = u, s, v, t
        else
            a, b, c, d = u, t, v, s
        end

        # skip if would create self-loop or multi-edge
        if a == b || c == d
            continue
        end
        if has_edge(g, a, b) || has_edge(g, c, d)
            continue
        end

        # perform the swap
        rem_edge!(g, u, v)
        rem_edge!(g, s, t)
        add_edge!(g, a, b)
        add_edge!(g, c, d)

        # rebuild edge list entry for these two slots
        edge_list[i] = Edge(min(a, b), max(a, b))
        edge_list[j] = Edge(min(c, d), max(c, d))
    end

    return g
end

function normalized_scores(graph::SimpleGraph, partitioner;
                           n_randomizations::Int=100,
                           rng::AbstractRNG=Xoshiro())
    part_orig = partitioner(graph)
    bp_orig = binarize_partition(graph, part_orig)
    Q_orig = modularity(graph, part_orig)
    rwc_orig = rwc(graph, bp_orig)

    Q_samples = Vector{Float64}(undef, n_randomizations)
    rwc_samples = Vector{Union{Missing, Float64}}(undef, n_randomizations)

    # pre-generate seeds for thread safety — each iteration gets its own RNG
    seeds = [rand(rng, UInt64) for _ in 1:n_randomizations]

    Threads.@threads for i in 1:n_randomizations
        thread_rng = Xoshiro(seeds[i])
        g_cm = configuration_model_sample(graph, thread_rng)
        part_cm = partitioner(g_cm)
        bp_cm = binarize_partition(g_cm, part_cm)
        Q_samples[i] = modularity(g_cm, part_cm)
        rwc_samples[i] = rwc(g_cm, bp_cm)
    end

    Q_valid = Q_samples
    rwc_valid = collect(skipmissing(rwc_samples))

    Q_norm = if std(Q_valid) > 0
        (Q_orig - mean(Q_valid)) / std(Q_valid)
    else
        0.0
    end

    rwc_norm = if !ismissing(rwc_orig) && length(rwc_valid) > 1 && std(rwc_valid) > 0
        (rwc_orig - mean(rwc_valid)) / std(rwc_valid)
    else
        missing
    end

    return (Q_norm=Q_norm, rwc_norm=rwc_norm)
end
