function bimodality_coefficient(x::AbstractVector{<:Real})
    n = length(x)
    if n < 3
        return 0.0
    end
    m = mean(x)
    s = std(x)
    if s == 0
        return 0.0
    end
    s3 = s^3
    s4 = s^4
    sum3 = 0.0
    sum4 = 0.0
    for xi in x
        d = xi - m
        d2 = d * d
        sum3 += d2 * d
        sum4 += d2 * d2
    end
    m3 = (sum3 / n) / s3
    m4 = (sum4 / n) / s4 - 3.0
    return (m3^2 + 1) / (m4 + 3 * (n - 1)^2 / ((n - 2) * (n - 3)))
end

function hartigan_dip(x::AbstractVector{<:Real}; min_is_zero::Bool=true)
    n = length(x)
    if n < 2
        return 0.0
    end

    xs = sort!(Float64.(collect(x)))
    if xs[end] == xs[1]
        return 0.0
    end

    mn = Vector{Int}(undef, n)
    mj = Vector{Int}(undef, n)
    gcm = Vector{Int}(undef, n)
    lcm = Vector{Int}(undef, n)

    mn[1] = 1
    for j in 2:n
        mn[j] = j - 1
        while true
            mnj = mn[j]
            mnmnj = mn[mnj]
            if mnj == 1 ||
               (xs[j] - xs[mnj]) * (mnj - mnmnj) < (xs[mnj] - xs[mnmnj]) * (j - mnj)
                break
            end
            mn[j] = mnmnj
        end
    end

    mj[n] = n
    for k in (n - 1):-1:1
        mj[k] = k + 1
        while true
            mjk = mj[k]
            mjmjk = mj[mjk]
            if mjk == n ||
               (xs[k] - xs[mjk]) * (mjk - mjmjk) < (xs[mjk] - xs[mjmjk]) * (k - mjk)
                break
            end
            mj[k] = mjmjk
        end
    end

    low = 1
    high = n
    dip2n = min_is_zero ? 0.0 : 1.0

    while true
        gcm[1] = high
        l_gcm = 1
        while gcm[l_gcm] > low
            l_gcm += 1
            gcm[l_gcm] = mn[gcm[l_gcm - 1]]
        end
        ig = l_gcm
        ix = max(ig - 1, 1)

        lcm[1] = low
        l_lcm = 1
        while lcm[l_lcm] < high
            l_lcm += 1
            lcm[l_lcm] = mj[lcm[l_lcm - 1]]
        end
        ih = l_lcm
        iv = 2

        d = 0.0
        if l_gcm != 2 || l_lcm != 2
            while true
                gcmix = gcm[ix]
                lcmiv = lcm[iv]
                if gcmix > lcmiv
                    gcmi1 = gcm[ix + 1]
                    denom = xs[gcmix] - xs[gcmi1]
                    dx = denom == 0.0 ? 0.0 :
                         (lcmiv - gcmi1 + 1) -
                         (xs[lcmiv] - xs[gcmi1]) * (gcmix - gcmi1) / denom
                    iv += 1
                    if dx >= d
                        d = dx
                        ig = ix + 1
                        ih = iv - 1
                    end
                else
                    lcmiv1 = lcm[iv - 1]
                    denom = xs[lcmiv] - xs[lcmiv1]
                    dx = denom == 0.0 ? 0.0 :
                         (xs[gcmix] - xs[lcmiv1]) * (lcmiv - lcmiv1) / denom -
                         (gcmix - lcmiv1 - 1)
                    ix -= 1
                    if dx >= d
                        d = dx
                        ig = ix + 1
                        ih = iv
                    end
                end
                if ix < 1
                    ix = 1
                end
                if iv > l_lcm
                    iv = l_lcm
                end
                if gcm[ix] == lcm[iv]
                    break
                end
            end
        else
            d = min_is_zero ? 0.0 : 1.0
        end

        if d < dip2n
            break
        end

        dip_l = 0.0
        for j in ig:(l_gcm - 1)
            max_t = 1.0
            jb = gcm[j + 1]
            je = gcm[j]
            if je - jb > 1 && xs[je] != xs[jb]
                c = (je - jb) / (xs[je] - xs[jb])
                for jj in jb:je
                    t = (jj - jb + 1) - (xs[jj] - xs[jb]) * c
                    if t > max_t
                        max_t = t
                    end
                end
            end
            if max_t > dip_l
                dip_l = max_t
            end
        end

        dip_u = 0.0
        for j in ih:(l_lcm - 1)
            max_t = 1.0
            jb = lcm[j]
            je = lcm[j + 1]
            if je - jb > 1 && xs[je] != xs[jb]
                c = (je - jb) / (xs[je] - xs[jb])
                for jj in jb:je
                    t = (xs[jj] - xs[jb]) * c - (jj - jb - 1)
                    if t > max_t
                        max_t = t
                    end
                end
            end
            if max_t > dip_u
                dip_u = max_t
            end
        end

        dipnew = max(dip_l, dip_u)
        if dipnew > dip2n
            dip2n = dipnew
        end

        new_low = gcm[ig]
        new_high = lcm[ih]
        if low == new_low && high == new_high
            break
        end
        low = new_low
        high = new_high
    end

    return dip2n / (2n)
end

function _silverman_bandwidth(x::Vector{Float64})
    n = length(x)
    if n < 2
        return 1.0
    end
    s = std(x)
    if s == 0.0
        return 1.0
    end
    q25, q75 = quantile(x, [0.25, 0.75])
    iqr_scale = (q75 - q25) / 1.34
    scale = iqr_scale > 0 ? min(s, iqr_scale) : s
    return 0.9 * scale * n^(-1 / 5)
end

function _gaussian_kde_grid(x::Vector{Float64}, grid::Vector{Float64}, h::Float64)
    n = length(x)
    dens = zeros(length(grid))
    inv = 1.0 / (n * h * sqrt(2.0 * pi))
    @inbounds for (i, gx) in enumerate(grid)
        acc = 0.0
        for xi in x
            z = (gx - xi) / h
            acc += exp(-0.5 * z * z)
        end
        dens[i] = inv * acc
    end
    return dens
end

function sample_valley_peak_ratio(x::AbstractVector{<:Real};
                                  grid_points::Int=512,
                                  bandwidth::Union{Nothing, Float64}=nothing,
                                  prominence_fraction::Float64=0.1)
    n = length(x)
    if n < 5
        return 1.0
    end

    xs = Float64.(collect(x))
    h = bandwidth === nothing ? _silverman_bandwidth(xs) : bandwidth
    if !isfinite(h) || h <= 0.0
        return 1.0
    end

    xmin = minimum(xs)
    xmax = maximum(xs)
    if xmin == xmax
        return 1.0
    end

    pad = 3.0 * h
    grid = collect(range(xmin - pad, xmax + pad, length=grid_points))
    dens = _gaussian_kde_grid(xs, grid, h)
    peak = maximum(dens)
    if peak <= 0.0
        return 1.0
    end

    mode_idx = Int[]
    for i in 2:(length(grid) - 1)
        if dens[i] >= dens[i - 1] && dens[i] > dens[i + 1] &&
           dens[i] >= prominence_fraction * peak
            push!(mode_idx, i)
        end
    end
    if length(mode_idx) < 2
        return 1.0
    end

    sort!(mode_idx, by=i -> dens[i], rev=true)
    i1, i2 = sort(mode_idx[1:2])
    valley = minimum(@view dens[i1:i2])
    peak2 = max(dens[i1], dens[i2])
    return clamp(valley / peak2, 0.0, 1.0)
end

function degree_opinion_correlation(graph::SimpleGraph, opinions::Vector{Float64},
                                    partition::Vector{Int})
    degs = degree(graph)
    abs_opinions = abs.(opinions)
    communities = sort(unique(partition))
    corrs = Float64[]
    for c in communities
        idx = findall(==(c), partition)
        if length(idx) < 3
            push!(corrs, 0.0)
        elseif std(degs[idx]) == 0.0 || std(abs_opinions[idx]) == 0.0
            push!(corrs, 0.0)
        else
            push!(corrs, cor(degs[idx], abs_opinions[idx]))
        end
    end
    return corrs
end
