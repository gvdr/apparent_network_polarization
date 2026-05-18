using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

println("=" ^ 60)
println("Running full experiment suite")
println("=" ^ 60)

t_start = time()

println()
println("--- Step 1: Parameter sweep ---")
include(joinpath(@__DIR__, "parameter_sweep.jl"))
t_sweep = time()
println("Sweep completed in " * string(round(t_sweep - t_start, digits=1)) * " seconds")

println()
println("--- Step 2: Model P calibration ---")
include(joinpath(@__DIR__, "model_p_calibration.jl"))
t_calib = time()
println("Calibration completed in " * string(round(t_calib - t_sweep, digits=1)) * " seconds")

println()
println("--- Step 3: Plotting ---")
include(joinpath(@__DIR__, "plot_results.jl"))
t_plot = time()
println("Plots completed in " * string(round(t_plot - t_calib, digits=1)) * " seconds")

println()
println("--- Step 4: Appendix H graph-metrics sweep ---")
include(joinpath(@__DIR__, "appendix_h_graph_metrics.jl"))
t_apph = time()
println("Appendix H sweep completed in " * string(round(t_apph - t_plot, digits=1)) * " seconds")

println()
println("=" ^ 60)
println("Total time: " * string(round(t_apph - t_start, digits=1)) * " seconds")
println("=" ^ 60)
