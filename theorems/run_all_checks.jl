using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

println("=" ^ 60)
println("Running all theorem checkers")
println("=" ^ 60)
println()

scripts = [
    "prop1_nonidentifiability.jl",
    "prop2_directional_distortion.jl",
    "prop2bis_monotonicity.jl",
    "prop3_bimodality_threshold.jl",
    "prop3_multifilter_subcase.jl",
    "prop3bis_oracle_separation.jl",
    "prop3ter_ei_monotonicity.jl",
    "prop4_selection_propagates.jl",
    "prop5_noise_preserves_bimodality.jl",
    "propB1_modularity_ei.jl",
    "corollary3_single_filter.jl",
    "closure_positive_feedback.jl",
    "normalization_limitations.jl",
]

results = Pair{String, Symbol}[]

for script in scripts
    path = joinpath(@__DIR__, script)
    println("-" ^ 60)
    println("Running: " * script)
    println("-" ^ 60)
    try
        verdict = include(path)
        if verdict === true
            push!(results, script => :PASS)
        else
            push!(results, script => :FAIL)
        end
    catch e
        println("ERROR: " * string(e))
        push!(results, script => :ERROR)
    end
    println()
end

println("=" ^ 60)
println("SUMMARY")
println("=" ^ 60)
for (script, status) in results
    println("  " * rpad(script, 45) * string(status))
end

n_pass = count(x -> x.second == :PASS, results)
n_total = length(results)
println()
println(string(n_pass) * " / " * string(n_total) * " checks passed")
