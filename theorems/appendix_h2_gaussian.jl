# Numerical check for Proposition H2.
# Verifies the identity ell(u,v) = log[cosh(mu(u+v)/sigma^2) /
# cosh(mu|u-v|/sigma^2)] for the two-topic mirrored Gaussian
# opportunity-structure model.
#
# Four layers of numerical checks:
#   1. Intermediate exponent expansion:
#      K_same(u,v) = 2 exp(-E) cosh(mu(u+v)/sigma^2)
#      K_opp(u,v)  = 2 exp(-E) cosh(mu|u-v|/sigma^2)
#      where E = (u^2 + v^2 + 2 mu^2) / (2 sigma^2).
#   2. Closed-form ell identity on a grid.
#   3. Conditions (A) ell >= 0 and (B) monotonicity.
#   4. Analytic vs numeric partial derivatives in u.
# Coverage: dense deterministic grid including u = 0, v = 0, u = v,
# plus 5000 random (mu, sigma, u, v) samples.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Statistics
using Random
using CairoMakie

println("=== Appendix H2: mirrored Gaussian identity ===")
println()

lambda_R(x, mu, sigma) = exp(-(x - mu)^2 / (2.0 * sigma^2))
lambda_L(x, mu, sigma) = exp(-(x + mu)^2 / (2.0 * sigma^2))

K_pair(x, y, mu, sigma) = lambda_R(x, mu, sigma) * lambda_R(y, mu, sigma) +
                          lambda_L(x, mu, sigma) * lambda_L(y, mu, sigma)

K_same_direct(u, v, mu, sigma) = (K_pair( u,  v, mu, sigma) + K_pair(-u, -v, mu, sigma)) / 2
K_opp_direct(u, v, mu, sigma)  = (K_pair( u, -v, mu, sigma) + K_pair(-u,  v, mu, sigma)) / 2

E_expon(u, v, mu, sigma)        = (u^2 + v^2 + 2.0 * mu^2) / (2.0 * sigma^2)
K_same_cosh(u, v, mu, sigma)    = 2.0 * exp(-E_expon(u, v, mu, sigma)) *
                                  cosh(mu * (u + v) / sigma^2)
K_opp_cosh(u, v, mu, sigma)     = 2.0 * exp(-E_expon(u, v, mu, sigma)) *
                                  cosh(mu * abs(u - v) / sigma^2)

ell_direct(u, v, mu, sigma)   = log(K_same_direct(u, v, mu, sigma) /
                                    K_opp_direct(u, v, mu, sigma))
ell_formula(u, v, mu, sigma)  = log(cosh(mu * (u + v)     / sigma^2) /
                                    cosh(mu * abs(u - v)  / sigma^2))

function d_ell_du_analytic(u, v, mu, sigma)
    t1 = tanh(mu * (u + v) / sigma^2)
    t2 = tanh(mu * abs(u - v) / sigma^2)
    s  = u > v ?  1.0 : (u < v ? -1.0 : 0.0)
    return (mu / sigma^2) * (t1 - s * t2)
end

function d_ell_du_numeric(u, v, mu, sigma; h = 1e-6)
    return (ell_formula(u + h, v, mu, sigma) - ell_formula(u - h, v, mu, sigma)) / (2 * h)
end

mu_values    = [0.25, 0.5, 1.0, 2.0, 3.0]
sigma_values = [0.25, 0.5, 1.0, 2.0, 3.0]
uv_grid      = collect(range(0.0, 5.0, length=21))

# --- Layer 1 & 2: exponent expansion and closed-form identity ---
tol_kernel = 1e-12
tol_ell    = 1e-12
max_err_same  = 0.0
max_err_opp   = 0.0
max_err_ell   = 0.0
for mu in mu_values, sigma in sigma_values, u in uv_grid, v in uv_grid
    global max_err_same = max(max_err_same, abs(K_same_direct(u,v,mu,sigma) - K_same_cosh(u,v,mu,sigma)))
    global max_err_opp  = max(max_err_opp,  abs(K_opp_direct(u,v,mu,sigma)  - K_opp_cosh(u,v,mu,sigma)))
    global max_err_ell  = max(max_err_ell,  abs(ell_direct(u,v,mu,sigma)    - ell_formula(u,v,mu,sigma)))
end
println("Intermediate identities (grid):")
println("  K_same: direct vs 2 e^{-E} cosh, max err = " * string(max_err_same))
println("  K_opp:  direct vs 2 e^{-E} cosh, max err = " * string(max_err_opp))
println("  ell = log(cosh(...) / cosh(...)), max err = " * string(max_err_ell))
pass_layer12 = max_err_same < tol_kernel && max_err_opp < tol_kernel && max_err_ell < tol_ell
println("  " * (pass_layer12 ? "PASS" : "FAIL"))
println()

# --- Layer 3: conditions (A) and (B) on grid ---
tol_cond = 1e-12
all_nonneg = true
for mu in mu_values, sigma in sigma_values, u in uv_grid, v in uv_grid
    if ell_formula(u, v, mu, sigma) < -tol_cond
        global all_nonneg = false
    end
end
println("(A) ell >= 0 on grid: " * (all_nonneg ? "PASS" : "FAIL"))

all_mono = true
for mu in mu_values, sigma in sigma_values
    for v in uv_grid
        row_u = [ell_formula(u, v, mu, sigma) for u in uv_grid]
        if any(diff(row_u) .< -tol_cond)
            global all_mono = false
        end
    end
    for u in uv_grid
        row_v = [ell_formula(u, v, mu, sigma) for v in uv_grid]
        if any(diff(row_v) .< -tol_cond)
            global all_mono = false
        end
    end
end
println("(B) monotone on grid: " * (all_mono ? "PASS" : "FAIL"))
println()

# --- Layer 4: analytic vs numeric partial derivative ---
# Skip |u - v| = 0 where d|u-v|/du is discontinuous, and the boundary
# points 0 and max where central differences step outside the tested grid.
tol_deriv = 1e-6
max_err_deriv = 0.0
n_deriv_checks = 0
for mu in mu_values, sigma in sigma_values,
    u in uv_grid[2:end-1], v in uv_grid[2:end-1]
    if abs(u - v) < 1e-3
        continue
    end
    d_num = d_ell_du_numeric(u, v, mu, sigma)
    d_ana = d_ell_du_analytic(u, v, mu, sigma)
    err = abs(d_num - d_ana)
    if err > max_err_deriv
        global max_err_deriv = err
    end
    global n_deriv_checks += 1
end
println("Derivative numeric vs analytic (" * string(n_deriv_checks) *
        " points, excluding u = v): max err = " *
        string(max_err_deriv) *
        " (" * (max_err_deriv < tol_deriv ? "PASS" : "FAIL") * ")")
println()

# --- Layer 5: random dense sampling ---
rng = Xoshiro(27182)
n_random = 5000
tol_rand = 1e-10
max_err_rand    = 0.0
mono_violations = 0
for _ in 1:n_random
    mu    = 0.1 + 3.0 * rand(rng)
    sigma = 0.25 + 3.0 * rand(rng)
    u     = 5.0 * rand(rng)
    v     = 5.0 * rand(rng)
    du    = 1e-3 + 1e-2 * rand(rng)
    err   = abs(ell_direct(u, v, mu, sigma) - ell_formula(u, v, mu, sigma))
    if err > max_err_rand
        global max_err_rand = err
    end
    if ell_formula(u + du, v, mu, sigma) + tol_rand < ell_formula(u, v, mu, sigma)
        global mono_violations += 1
    end
end
println("Random dense sampling (n = " * string(n_random) * "):")
println("  closed-form identity, max err = " * string(max_err_rand))
println("  monotone-in-u violations       = " * string(mono_violations))
pass_rand = max_err_rand < tol_rand && mono_violations == 0
println("  " * (pass_rand ? "PASS" : "FAIL"))
println()

verdict = pass_layer12 && all_nonneg && all_mono &&
          (max_err_deriv < tol_deriv) && pass_rand
println((verdict ? "OVERALL PASS" : "OVERALL FAIL") *
        " (alg " * string(max(max_err_same, max_err_opp, max_err_ell)) *
        ", deriv " * string(max_err_deriv) *
        ", rand " * string(max_err_rand) * ")")

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(700, 450))
mu_plot    = 1.0
sigma_plot = 1.0
u_fine = collect(range(0.0, 3.0, length=41))
v_fine = collect(range(0.0, 3.0, length=41))
ell_mat = [ell_formula(u, v, mu_plot, sigma_plot) for u in u_fine, v in v_fine]
ax = Axis(fig[1, 1], xlabel="u", ylabel="v",
          title="Mirrored Gaussian log-odds ell(u,v) at mu = " *
                string(mu_plot) * ", sigma = " * string(sigma_plot))
hm = heatmap!(ax, u_fine, v_fine, ell_mat, colormap=:viridis)
Colorbar(fig[1, 2], hm)
save(joinpath(@__DIR__, "..", "output", "theorems", "appendix_h2_gaussian.png"), fig)
println("Figure saved.")
verdict
