# --- Partitioning ---

function oracle_sign_partition(opinions::Vector{Float64})
    return [x >= 0.0 ? 2 : 1 for x in opinions]
end

function community_partition(graph::SimpleGraph)
    if nv(graph) == 0
        return Int[]
    end
    return Graphs.louvain(graph)
end

function binarize_partition(graph::SimpleGraph, partition::Vector{Int})
    if isempty(partition)
        return Int[]
    end
    community_ids = sort(unique(partition))
    if length(community_ids) <= 2
        if length(community_ids) == 1
            return ones(Int, length(partition))
        end
        label_map = Dict(community_ids[1] => 1, community_ids[2] => 2)
        return [label_map[c] for c in partition]
    end

    # find two largest communities by node count
    counts = Dict{Int, Int}()
    for c in partition
        counts[c] = get(counts, c, 0) + 1
    end
    sorted_comms = sort(collect(counts), by=x -> (-x.second, x.first))
    side_a = sorted_comms[1].first
    side_b = sorted_comms[2].first

    result = zeros(Int, length(partition))
    for i in eachindex(partition)
        c = partition[i]
        if c == side_a
            result[i] = 1
        elseif c == side_b
            result[i] = 2
        end
    end

    # merge remaining communities based on edge connectivity
    remaining_comms = setdiff(Set(community_ids), Set([side_a, side_b]))
    for rc in remaining_comms
        rc_nodes = findall(==(rc), partition)
        edges_to_a = 0
        edges_to_b = 0
        for node in rc_nodes
            for neighbor in neighbors(graph, node)
                if partition[neighbor] == side_a
                    edges_to_a += 1
                elseif partition[neighbor] == side_b
                    edges_to_b += 1
                end
            end
        end
        merge_to = edges_to_a >= edges_to_b ? 1 : 2
        for node in rc_nodes
            result[node] = merge_to
        end
    end

    return result
end

# --- Metrics ---

# modularity: delegate to Graphs.jl
const modularity = Graphs.modularity

function ei_index(graph::SimpleGraph, binary_partition::Vector{Int})
    e_int = 0
    e_ext = 0
    for e in edges(graph)
        if binary_partition[src(e)] == binary_partition[dst(e)]
            e_int += 1
        else
            e_ext += 1
        end
    end
    total = e_int + e_ext
    if total == 0
        return 0.0
    end
    return (e_ext - e_int) / total
end

function spectral_gap(graph::SimpleGraph)
    n = nv(graph)
    if n <= 1
        return 0.0
    end
    A_sparse = adjacency_matrix(graph)
    degs = vec(sum(A_sparse, dims=2))
    d_inv_sqrt = zeros(n)
    for i in 1:n
        if degs[i] > 0
            d_inv_sqrt[i] = 1.0 / sqrt(Float64(degs[i]))
        end
    end
    D_inv_sqrt = Diagonal(d_inv_sqrt)
    # Keep sparse until the final eigen call
    L_norm_sparse = sparse(I, n, n) - D_inv_sqrt * A_sparse * D_inv_sqrt
    # For N=500, dense eigen is still fast (~50ms). For larger N, use iterative methods.
    evals = eigvals(Symmetric(Matrix(L_norm_sparse)))
    sort!(evals)
    return evals[2]
end

function rwc(graph::SimpleGraph, binary_partition::Vector{Int}; k::Int=10)
    side1 = findall(==(1), binary_partition)
    side2 = findall(==(2), binary_partition)

    if length(side1) < 2 || length(side2) < 2
        return missing
    end

    k1 = min(k, length(side1))
    k2 = min(k, length(side2))

    degs = degree(graph)
    absorbers1 = side1[partialsortperm(degs[side1], 1:k1, rev=true)]
    absorbers2 = side2[partialsortperm(degs[side2], 1:k2, rev=true)]

    is_absorber = falses(nv(graph))
    is_absorber1 = falses(nv(graph))
    for v in absorbers1
        is_absorber[v] = true
        is_absorber1[v] = true
    end
    for v in absorbers2
        is_absorber[v] = true
    end
    transient = [v for v in 1:nv(graph) if !is_absorber[v]]

    if isempty(transient)
        return missing
    end

    n_trans = length(transient)
    trans_idx = Dict(v => i for (i, v) in enumerate(transient))

    Q_mat = zeros(n_trans, n_trans)
    R1 = zeros(n_trans)
    R2 = zeros(n_trans)

    for (i, v) in enumerate(transient)
        d = degs[v]
        if d == 0
            continue
        end
        for u in neighbors(graph, v)
            if is_absorber[u]
                if is_absorber1[u]
                    R1[i] += 1.0 / d
                else
                    R2[i] += 1.0 / d
                end
            else
                j = trans_idx[u]
                Q_mat[i, j] = 1.0 / d
            end
        end
    end

    IQ = I - Q_mat
    local B1, B2
    try
        F = lu(IQ)
        B1 = F \ R1
        B2 = F \ R2
    catch e
        if e isa LinearAlgebra.SingularException
            return missing
        end
        rethrow(e)
    end

    side1_trans = [trans_idx[v] for v in transient if binary_partition[v] == 1]
    side2_trans = [trans_idx[v] for v in transient if binary_partition[v] == 2]

    if isempty(side1_trans) || isempty(side2_trans)
        return missing
    end

    P_XX = mean(B1[side1_trans])
    P_YY = mean(B2[side2_trans])
    P_XY = mean(B2[side1_trans])
    P_YX = mean(B1[side2_trans])

    return P_XX * P_YY - P_XY * P_YX
end

function compute_all_metrics(graph::SimpleGraph, partition::Vector{Int},
                             binary_partition::Vector{Int}; k::Int=10)
    return MetricsTuple((
        modularity(graph, partition),
        rwc(graph, binary_partition; k=k),
        spectral_gap(graph),
        ei_index(graph, binary_partition),
    ))
end
