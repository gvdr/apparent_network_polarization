# PolarizationPipeline.jl

Simulation package for "Apparent Polarization in Social Networks: Heterogeneous Activity and the Distortion of Structural Metrics."

This package implements the four-stage measurement pipeline (opinion distribution, network formation, observation filter, metric computation) and reproduces the manuscript's theoretical checks and numerical experiments.

## Installation

Requires Julia 1.11+.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

```julia
using PolarizationPipeline

config = manuscript_config_A(N=1000, α=2.0, β=1.0, τ=0.5)
result = run_pipeline(config)
result.metrics  # (Q, rwc, spectral_gap, ei)
```

Three preset configurations match the manuscript scenarios:

- `manuscript_config_A` -- apolitical population (unimodal opinions, activity bias)
- `manuscript_config_P` -- polarized population (bimodal opinions)
- `linear_homophily_config` -- uniform opinions with linear homophily

## Repository structure

```
src/            Julia package source
test/           Unit tests
theorems/       Numerical verification of each proposition
experiments/    Parameter sweeps and manuscript figure scripts
output/         Pre-generated figures and data
```

## Running

```bash
# Unit tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Verify a single proposition
julia --project=. theorems/prop3_bimodality_threshold.jl

# Run all theorem checks
julia --project=. theorems/run_all_checks.jl

# Full simulation suite
julia --project=. experiments/run_all.jl
```

## License

See [LICENSE](LICENSE).
