function observed_density(x::Float64, f, phi; Z::Float64=1.0)
    return f(x) * phi(x) / Z
end

function find_modes(density_func; grid_range::Tuple{Float64,Float64}=(-5.0, 5.0),
                    grid_points::Int=10000)
    xs = range(grid_range[1], grid_range[2], length=grid_points)
    ys = [density_func(x) for x in xs]
    modes = Float64[]
    for i in 2:(grid_points-1)
        if ys[i] >= ys[i-1] && ys[i] > ys[i+1]
            push!(modes, xs[i])
        end
    end
    return modes
end

function valley_to_peak_ratio(density_func, modes::Vector{Float64})
    if length(modes) < 2
        return 1.0
    end
    m1, m2 = extrema(modes)
    xs = range(m1, m2, length=1000)
    valley_val = minimum(density_func(x) for x in xs)
    peak_val = maximum(density_func(m) for m in modes)
    if peak_val == 0.0
        return 1.0
    end
    return valley_val / peak_val
end

function heteroskedastic_convolution(f_obs, noise_scale_func;
                                     grid_range::Tuple{Float64,Float64}=(-6.0, 6.0),
                                     grid_points::Int=1000,
                                     integration_points::Int=500)
    zs = collect(range(grid_range[1], grid_range[2], length=grid_points))
    xs = range(grid_range[1], grid_range[2], length=integration_points)
    dx = step(xs)
    result = zeros(grid_points)
    inv_sqrt_2pi = 1.0 / sqrt(2.0 * pi)
    for (iz, z) in enumerate(zs)
        val = 0.0
        for x in xs
            sigma_x = noise_scale_func(x)
            if sigma_x > 0
                diff = z - x
                val += f_obs(x) * inv_sqrt_2pi / sigma_x * exp(-0.5 * (diff / sigma_x)^2) * dx
            end
        end
        result[iz] = val
    end
    return result
end
