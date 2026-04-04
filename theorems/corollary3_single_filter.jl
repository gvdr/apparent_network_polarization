using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Distributions

println("=== Corollary 3: Single Filter Sufficiency ===")
println()

sigma = 1.0
threshold = 1.0 / (2.0 * sigma^2)

# r_k = phi_k(sigma) / phi_k(0) = 1 + phi_1k * sigma^2 / phi_0k
# so phi_1k / phi_0k = (r_k - 1) / sigma^2

# Test 1: single filter with r_k = 1.6 (> 1.5)
r1 = 1.6
ratio1 = (r1 - 1.0) / sigma^2
density1 = x -> exp(-x^2 / (2.0 * sigma^2)) * (1.0 + ratio1 * x^2)
modes1 = find_modes(density1; grid_range=(-5.0, 5.0), grid_points=20000)
println("Single filter r_k = " * string(r1) * ": ratio = " * string(ratio1) * ", modes = " * string(length(modes1)))
test1 = length(modes1) == 2

# Test 2: single filter with r_k = 1.3 (< 1.5)
r2 = 1.3
ratio2 = (r2 - 1.0) / sigma^2
density2 = x -> exp(-x^2 / (2.0 * sigma^2)) * (1.0 + ratio2 * x^2)
modes2 = find_modes(density2; grid_range=(-5.0, 5.0), grid_points=20000)
println("Single filter r_k = " * string(r2) * ": ratio = " * string(ratio2) * ", modes = " * string(length(modes2)))
test2 = length(modes2) == 1

# Test 3: 4 filters with r_k = 1.15 each
r3 = 1.15
ratio3 = (r3 - 1.0) / sigma^2
sum_ratios = 4 * ratio3
println("4 filters r_k = " * string(r3) * " each: sum of ratios = " * string(sum_ratios) * " (threshold = " * string(threshold) * ")")
density3 = x -> begin
    val = exp(-x^2 / (2.0 * sigma^2))
    for _ in 1:4
        val *= (1.0 + ratio3 * x^2)
    end
    val
end
modes3 = find_modes(density3; grid_range=(-5.0, 5.0), grid_points=20000)
println("4 filters jointly: modes = " * string(length(modes3)))
test3 = length(modes3) == 2

verdict = test1 && test2 && test3
if verdict
    println("PASS")
else
    println("FAIL: test1=" * string(test1) * " test2=" * string(test2) * " test3=" * string(test3))
end
verdict
