# Numerical check for Proposition H1.
# The analytic identity ell(u,v) = 2*artanh(r(u)*r(v)) is proved in the
# manuscript for any K(x,y) = a(x)*a(y). This script is family-specific
# support on a finite (u,v) grid, under a(x) = 1 + tanh(alpha*x).
#
# The script runs four layers of numerical checks:
#   1. Intermediate algebraic identities (K_same/K_opp via s/p decomposition,
#      closed-form of ell in terms of r).
#   2. Closed-form identity ell = 2*artanh(r(u)*r(v)).
#   3. Conditions (A) ell >= 0 and (B) monotonicity in each coordinate.
#   4. Numeric vs analytic first-order derivative.
# Coverage: a dense deterministic grid including boundary points u=0, v=0,
# and u=v, plus a block of random (alpha, u, v) sampling.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using PolarizationPipeline
using Statistics
using Random
using CairoMakie

println("=== Appendix H1: single-topic identity (a(x) = 1 + tanh(alpha*x)) ===")
println()

a_fn(x, alpha)  = 1.0 + tanh(alpha * x)
r_fn(u, alpha)  = tanh(alpha * u)
rp_fn(u, alpha) = alpha * (1.0 - tanh(alpha * u)^2)   # r'(u)

function K_same_direct(u, v, alpha)
    au  = a_fn(u, alpha);  amu = a_fn(-u, alpha)
    av  = a_fn(v, alpha);  amv = a_fn(-v, alpha)
    return (au * av + amu * amv) / 2
end

function K_opp_direct(u, v, alpha)
    au  = a_fn(u, alpha);  amu = a_fn(-u, alpha)
    av  = a_fn(v, alpha);  amv = a_fn(-v, alpha)
    return (au * amv + amu * av) / 2
end

function K_same_sp(u, v, alpha)
    s_u = a_fn(u, alpha) + a_fn(-u, alpha)
    p_u = a_fn(u, alpha) - a_fn(-u, alpha)
    s_v = a_fn(v, alpha) + a_fn(-v, alpha)
    p_v = a_fn(v, alpha) - a_fn(-v, alpha)
    return (s_u * s_v + p_u * p_v) / 4
end

function K_opp_sp(u, v, alpha)
    s_u = a_fn(u, alpha) + a_fn(-u, alpha)
    p_u = a_fn(u, alpha) - a_fn(-u, alpha)
    s_v = a_fn(v, alpha) + a_fn(-v, alpha)
    p_v = a_fn(v, alpha) - a_fn(-v, alpha)
    return (s_u * s_v - p_u * p_v) / 4
end

ell_direct(u, v, alpha)  = log(K_same_direct(u, v, alpha) / K_opp_direct(u, v, alpha))
ell_r_form(u, v, alpha)  = log((1.0 + r_fn(u, alpha) * r_fn(v, alpha)) /
                                (1.0 - r_fn(u, alpha) * r_fn(v, alpha)))
ell_atanh(u, v, alpha)   = 2.0 * atanh(r_fn(u, alpha) * r_fn(v, alpha))

function d_ell_du_analytic(u, v, alpha)
    ru  = r_fn(u, alpha)
    rv  = r_fn(v, alpha)
    rpu = rp_fn(u, alpha)
    return 2.0 * rpu * rv / (1.0 - (ru * rv)^2)
end

function d_ell_du_numeric(u, v, alpha; h = 1e-6)
    return (ell_atanh(u + h, v, alpha) - ell_atanh(u - h, v, alpha)) / (2 * h)
end

alpha_values = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
uv_grid      = collect(range(0.0, 5.0, length=21))

# --- Layer 1 & 2: intermediate identities and closed form ---
# Skip points where |r(u)*r(v)| saturates within 1e-12 of 1 (catastrophic
# cancellation in log((1+rr)/(1-rr)) is a numerical artefact, not a
# violation of the identity).
# Two tolerances: the K_same / K_opp decomposition is pure arithmetic and
# expected at machine precision; the ell formulas involve log and atanh
# near saturation, which lose several digits when |rr| approaches 1.
tol_alg_kernel = 1e-12    # K_same, K_opp identities
tol_alg_ell    = 1e-7     # ell closed-form identities
sat_tol        = 1e-12
max_err_sp       = 0.0    # K_same vs s/p form
max_err_sp_opp   = 0.0    # K_opp vs s/p form
max_err_r_form   = 0.0    # log((1+rr)/(1-rr)) vs direct
max_err_atanh    = 0.0    # 2*atanh(rr) vs direct
for alpha in alpha_values, u in uv_grid, v in uv_grid
    global max_err_sp     = max(max_err_sp,     abs(K_same_direct(u,v,alpha) - K_same_sp(u,v,alpha)))
    global max_err_sp_opp = max(max_err_sp_opp, abs(K_opp_direct(u,v,alpha)  - K_opp_sp(u,v,alpha)))
    rr = r_fn(u, alpha) * r_fn(v, alpha)
    if abs(rr) < 1 - sat_tol
        global max_err_r_form = max(max_err_r_form, abs(ell_direct(u,v,alpha) - ell_r_form(u,v,alpha)))
        global max_err_atanh  = max(max_err_atanh,  abs(ell_direct(u,v,alpha) - ell_atanh(u,v,alpha)))
    end
end
println("Intermediate algebraic identities (grid):")
println("  K_same: direct vs s/p form, max err = " * string(max_err_sp))
println("  K_opp:  direct vs s/p form, max err = " * string(max_err_sp_opp))
println("  ell = log((1 + r r) / (1 - r r)),  max err = " * string(max_err_r_form))
println("  ell = 2 * atanh(r r),                max err = " * string(max_err_atanh))
pass_kernel = max(max_err_sp, max_err_sp_opp) < tol_alg_kernel
pass_ell    = max(max_err_r_form, max_err_atanh) < tol_alg_ell
pass_layer12 = pass_kernel && pass_ell
println("  " * (pass_layer12 ? "PASS" : "FAIL"))
println()

# --- Layer 3: conditions (A) and (B) on an expanded deterministic grid ---
tol_cond = 1e-10
all_nonneg = true
for alpha in alpha_values, u in uv_grid, v in uv_grid
    if ell_atanh(u, v, alpha) < -tol_cond
        global all_nonneg = false
    end
end
println("(A) ell >= 0 on grid: " * (all_nonneg ? "PASS" : "FAIL"))

all_mono = true
for alpha in alpha_values
    for v in uv_grid
        row_u = [ell_atanh(u, v, alpha) for u in uv_grid]
        if any(diff(row_u) .< -tol_cond)
            global all_mono = false
        end
    end
    for u in uv_grid
        row_v = [ell_atanh(u, v, alpha) for v in uv_grid]
        if any(diff(row_v) .< -tol_cond)
            global all_mono = false
        end
    end
end
println("(B) monotone on grid: " * (all_mono ? "PASS" : "FAIL"))
println()

# --- Layer 4: derivative check (numeric vs analytic) ---
# Restricted to non-saturation regime: |r(u) r(v)| < 0.99, so that
# 1/(1 - (rr)^2) stays bounded and finite differences are well-conditioned.
tol_deriv = 1e-6
max_err_deriv = 0.0
n_deriv_checks = 0
for alpha in [0.1, 0.3, 0.5, 1.0], u in uv_grid[2:end-1], v in uv_grid[2:end-1]
    rr = r_fn(u, alpha) * r_fn(v, alpha)
    if abs(rr) >= 0.99
        continue
    end
    d_num = d_ell_du_numeric(u, v, alpha)
    d_ana = d_ell_du_analytic(u, v, alpha)
    err = abs(d_num - d_ana)
    if err > max_err_deriv
        global max_err_deriv = err
    end
    global n_deriv_checks += 1
end
println("Derivative numeric vs analytic (" * string(n_deriv_checks) *
        " non-saturation points): max err = " * string(max_err_deriv) *
        " (" * (max_err_deriv < tol_deriv ? "PASS" : "FAIL") * ")")
println()

# --- Random dense sampling: closed-form identity and monotonicity in u ---
rng = Xoshiro(31415)
n_random = 5000
tol_rand = 1e-7
max_err_rand    = 0.0
mono_violations = 0
for _ in 1:n_random
    alpha = rand(rng, [0.0, 0.1, 0.3, 0.5, 1.0, 2.0])
    u  = 5.0 * rand(rng)
    v  = 5.0 * rand(rng)
    du = 1e-3 + 1e-2 * rand(rng)
    rr = r_fn(u, alpha) * r_fn(v, alpha)
    if abs(rr) < 1 - sat_tol
        err = abs(ell_direct(u, v, alpha) - ell_atanh(u, v, alpha))
        if err > max_err_rand
            global max_err_rand = err
        end
    end
    if ell_atanh(u + du, v, alpha) + tol_rand < ell_atanh(u, v, alpha)
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
        " (alg " * string(max(max_err_sp, max_err_sp_opp, max_err_r_form, max_err_atanh)) *
        ", deriv " * string(max_err_deriv) *
        ", rand " * string(max_err_rand) * ")")

mkpath(joinpath(@__DIR__, "..", "output", "theorems"))
fig = Figure(size=(700, 450))
alpha_plot = 1.0
u_fine = collect(range(0.0, 3.0, length=41))
v_fine = collect(range(0.0, 3.0, length=41))
ell_mat = [ell_atanh(u, v, alpha_plot) for u in u_fine, v in v_fine]
ax = Axis(fig[1, 1], xlabel="u", ylabel="v",
          title="Single-topic log-odds ell(u,v) at alpha = " * string(alpha_plot))
hm = heatmap!(ax, u_fine, v_fine, ell_mat, colormap=:viridis)
Colorbar(fig[1, 2], hm)
save(joinpath(@__DIR__, "..", "output", "theorems", "appendix_h1_single_topic.png"), fig)
println("Figure saved.")
verdict
