using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

println("=" ^ 60)
println("Running theorem stress suite")
println("=" ^ 60)
println()

scripts = [
    "stress_prop3_boundary.jl",
    "stress_prop3bis_randomized.jl",
    "stress_prop3ter_b1_randomized.jl",
    "stress_prop4_replications.jl",
    "stress_prop5_boundary.jl",
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
println("STRESS SUMMARY")
println("=" ^ 60)
for (script, status) in results
    println("  " * rpad(script, 40) * string(status))
end

n_pass = count(x -> x.second == :PASS, results)
n_total = length(results)
println()
println(string(n_pass) * " / " * string(n_total) * " stress checks passed")
