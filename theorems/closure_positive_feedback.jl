using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Random

println("=== Closure Under Positive Feedback ===")
println()

rng = Xoshiro(42)
test_points = range(-3.0, 3.0, length=100)
n_trials = 50
all_pass = true

for trial in 1:n_trials
    # random extremism-amplifying functions: phi(x) = a0 + a1*x^2 with a0>0, a1>=0
    a0_1 = 0.5 + rand(rng)
    a1_1 = rand(rng) * 3.0
    a0_2 = 0.5 + rand(rng)
    a1_2 = rand(rng) * 3.0

    local phi1 = x -> a0_1 + a1_1 * x^2
    local phi2 = x -> a0_2 + a1_2 * x^2
    local product = x -> phi1(x) * phi2(x)

    # check: product is non-decreasing in |x|
    vals = [product(x) for x in test_points]
    abs_vals = abs.(collect(test_points))
    # for |x| increasing from 0, product should be non-decreasing
    pos_points = filter(x -> x >= 0.0, collect(test_points))
    pos_vals = [product(x) for x in pos_points]
    if !all(diff(pos_vals) .>= -1e-10)
        println("FAIL at trial " * string(trial) * ": product not non-decreasing in |x|")
        global all_pass = false
    end

    # check: product(x) >= product(0) for all x (minimum at origin)
    p0 = product(0.0)
    if !all(v >= p0 - 1e-10 for v in vals)
        println("FAIL at trial " * string(trial) * ": product not minimized at origin")
        global all_pass = false
    end
end

println("Tested " * string(n_trials) * " random (phi1, phi2) pairs")
verdict = all_pass
if verdict
    println("PASS: product is always extremism-amplifying")
else
    println("FAIL")
end
verdict
