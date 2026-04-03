# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
julia --project=. test/runtests.jl

# Install / update dependencies (also adds Test if missing)
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.add("Test")'

# Run end-to-end examples (simulate data, fit model, print results)
julia --project=. examples/ex1_warfarin.jl
julia --project=. examples/ex2_two_cpt_iv.jl
julia --project=. examples/ex3_two_cpt_oral_cov.jl
julia --project=. examples/ex4_ode_mm.jl          # ODE-based MM elimination

# Start Julia REPL with package loaded
julia --project=. -e 'using JuliaNLME'
```

## Architecture

JuliaNLME is a Non-Linear Mixed Effects (NLME) modeling library for pharmacokinetics using the FOCE (First-Order Conditional Estimation) method.

### Layer structure (load order matters)

```
src/types.jl                     — Core structs (Subject, Population, ModelParameters, FitResult, ODESpec, etc.)
src/pk/one_compartment.jl        — 1-CMT analytical equations + _one_cpt_single_dose_fn
src/pk/two_compartment.jl        — 2-CMT analytical equations + _two_cpt_single_dose_fn
src/pk/ode_solver.jl             — _ode_predictions: segment-by-segment ODE integration with dose events
src/JuliaNLME.jl                 — make_single_dose_fn dispatcher (after both PK files are loaded)
src/stats/residual_error.jl      — residual_variance, compute_R_diag, iwres, cwres
src/stats/likelihood.jl          — individual_nll (inner loop obj), foce_subject_nll, foce_population_nll
src/estimation/parameterization.jl — pack_params/unpack_params (Cholesky/log transforms)
src/estimation/inner_optimizer.jl  — find_ebe: per-subject η optimization via ForwardDiff + Optim
src/estimation/outer_optimizer.jl  — optimize_population: BFGS over θ/Ω/σ + covariance step
src/estimation/saem.jl             — SAEM estimation (stochastic EM, MH sampling)
src/estimation/its.jl              — ITS estimation (deterministic two-stage, MAP + closed-form Ω)
src/estimation/importance_sampling.jl — IS parameter uncertainty
src/io/datareader.jl             — read_data: NONMEM-format CSV → Population
src/io/output.jl                 — print_results, parameter_table, sdtab
src/parser/model_parser.jl       — parse .jnlme model files → CompiledModel
src/api.jl                       — Public fit(), simulate()
```

### Estimation flow

**Outer loop** (BFGS over packed θ/Ω/σ):
1. Unpack parameters via Cholesky/log transforms (`parameterization.jl`)
2. **Inner loop** (per-subject, parallelized): minimize `individual_nll(η | θ,Ω,σ)` via ForwardDiff gradient → returns `η̂ᵢ` and Jacobian `Hᵢ = ∂f/∂η|_{η̂ᵢ}`
3. Compute FOCE population OFV: `Σᵢ foce_subject_nll(η̂ᵢ, Hᵢ, ...)`
4. Return OFV to BFGS

After convergence: covariance step via `ForwardDiff.hessian` of OFV.

### AD compatibility — critical invariant

**Every function in the PK/likelihood call chain must be generic over `T<:Real`**, not hardcoded to `Float64`. ForwardDiff passes `Dual{...}` numbers through the computation graph.

- PK functions use multiple type parameters + `promote_type` (e.g., `cl::C, v::V, t::TT where {C,V,TT<:Real}`)
- Closures in `_one_cpt_single_dose_fn` / `_two_cpt_single_dose_fn` capture concrete `Float64` values; the called `_impl` function handles promotion
- Never use `if` with type-dispatch in the hot PK path — use `ifelse` for AD-compatible branching

### Ω parameterization

Ω is stored as its Cholesky factor `L` (lower-triangular). For optimization: diagonal elements stored as `log(Lᵢᵢ)`, off-diagonal as-is. Use `C_Ω \ eta` (triangular solve) everywhere — never `inv(Ω)`.

### Model file format (`.jnlme`)

**Analytical model** (uses built-in PK equations):
```
model Name
  [parameters]    — theta NAME(init, lower, upper) / omega ETA ~ var / sigma NAME ~ var
  [individual_parameters]  — plain Julia math; variables are theta/eta names + covariate names
  [structural_model]       — pk model_symbol(cl=CL, v=V, ...)
  [error_model]            — DV ~ proportional|additive|combined(SIGMA_NAME)
end
```

**ODE model** (arbitrary differential equations):
```
model Name
  [parameters]    — same as above
  [individual_parameters]  — defines PK params used in ODEs (ALL defined vars are passed to ODE as p.NAME)
  [structural_model]       — ode(obs_cmt=STATE_NAME, states=[STATE1, STATE2, ...])
  [odes]                   — d/dt(STATE) = expression  (one line per state)
  [error_model]            — same as above
end
```

The `[individual_parameters]` block is parsed via `Meta.parse` and compiled via `RuntimeGeneratedFunctions.jl` to avoid world-age issues. For ODE models, the `pk_param_fn` returns ALL defined variables (not just mapped ones), and these are accessible in the ODE as `p.NAME`.

### ODE model design

- **`ODESpec`** (in `types.jl`): holds the compiled ODE function, state name list, and `obs_cmt_idx`
- **Dose handling**: `_ode_predictions` (in `ode_solver.jl`) integrates segment-by-segment, applying bolus doses as instantaneous state increments at each dose time. Only bolus doses (`rate==0`) are supported.
- **ForwardDiff**: `u0 = zeros(T, n_states)` where T is inferred from `typeof(first(values(pk_params)))`. Since `Tsit5` is a pure explicit method, Dual-typed states propagate through automatically.
- **Solver**: `Tsit5()` (explicit, 5th-order RK). Suitable for most non-stiff PK ODE systems.
- **Observation extraction**: uses `saveat` to evaluate the ODE at each obs time. No callbacks needed.

### Supported PK models

| Symbol | Required params |
|---|---|
| `:one_cpt_iv_bolus` | `cl, v` |
| `:one_cpt_infusion` | `cl, v` |
| `:one_cpt_oral` | `cl, v, ka` (optional: `f`) |
| `:two_cpt_iv_bolus` | `cl, v1, q, v2` |
| `:two_cpt_infusion` | `cl, v1, q, v2` |
| `:two_cpt_oral` | `cl, v1, q, v2, ka` (optional: `f`) |

### Outer optimizer design

`make_outer_objective` returns `(f_only, g_only!, fdfg!, state)` for use with `OnceDifferentiable(f, g!, fg!, x0)`. The combined `fdfg!(G, x)` runs the inner loop **once** and uses the same EBEs for both the OFV value and the ForwardDiff gradient — this consistency is critical for BFGS convergence. Separate `obj_fn` / `grad_fn!` with independent inner loops caused inconsistent (f, g) pairs and poor convergence.

The gradient uses `foce_population_nll_diff` with EBEs held fixed (standard FOCE approach). This differentiates only through the likelihood computation, not through the inner EBE optimization.

### Numerical robustness

- **`OmegaMatrix` constructor**: tries standard cholesky; regularizes with `λ_min + 1e-8` if near-singular (handles extreme optimizer steps)
- **`foce_subject_nll_raw`**: checks `issuccess` on both R_tilde and Ω choleskeys; returns `T(1e20)` on failure so ForwardDiff produces zero gradient (not NaN)
- **Inner optimizer**: uses `BackTracking` line search (not `HagerZhang`) and wraps each subject's optimization in try-catch
- **NaN gradient warnings** from Optim during early BFGS iterations are expected — they occur when the line search explores extreme parameter regions, Optim recovers and continues

### Key design decisions

- **Singularity handling**: KA≈k in oral models handled via `ifelse` smooth switch (not `if`) for AD safety
- **Column names**: `datareader.jl` normalizes all columns to lowercase Symbols. Use `propertynames(df)` not `names(df)` for Symbol membership checks (DataFrames `names()` returns `Vector{String}`)
- **Named tuple AST**: return expressions in parser codegen use `Expr(:(=), key, val)` not `Expr(:kw, ...)`
- **Covariate name casing**: `read_data` stores covariate keys as lowercase Symbols (`:wt`, `:crcl`). The parser detects uppercase identifiers in `[individual_parameters]` as covariates (e.g., `WT`, `CRCL`) and injects `WT = covariates[:wt]` — i.e., the lookup key is lowercased. Model files use uppercase names; dict keys are always lowercase.
- **`sdtab` ETA columns**: named `ETA1`, `ETA2`, ... (positional, not by ETA name). When building per-subject diagnostic DataFrames, access ETAs directly via `result.subjects[i].eta[k]` rather than parsing the sdtab output.
- **FOCE vs FOCEI objectives** (`foce_subject_nll` in `likelihood.jl`):
  - `interaction=false` (FOCE, paper eq. 15): `0.5×[(y-f0)ᵀR̃⁻¹(y-f0) + log|R̃|]`. Ω enters only through R̃=HΩHᵀ+R; no Ω collapse risk.
  - `interaction=true` (FOCEI, paper eq. 20 simplified via matrix determinant lemma): `0.5×[(y-IPRED)ᵀV⁻¹(y-IPRED) + η̂ᵀΩ⁻¹η̂ + log|R̃|]`. The three Ω terms in eq. 20 reduce to just `log|R̃|` plus `η̂ᵀΩ⁻¹η̂`; the `log|Ω|` and `log|V|` cancel. The `η̂ᵀΩ⁻¹η̂` term prevents Ω→0 (pushing it to +∞), while `log|R̃|` grows if Ω→∞.
- **NONMEM OFV convention**: `OFV = 2×NLL` (no `n_obs×log(2π)` constant). NONMEM omits this constant from its reported OFV since it is a constant w.r.t. parameters and cancels in model comparisons. The `n_eta×log(2π)` terms from the η prior and Laplace correction cancel each other exactly.
- **No EM post-processing**: the FOCEI objective is inherently well-conditioned for Ω. There is no separate EM M-step.
- **ITS Ĉᵢ formula**: `compute_C_hat` computes `[HᵀV⁻¹H + Ω⁻¹]⁻¹` where V = diag(R_diag) and Ω⁻¹ = L⁻ᵀL⁻¹. This corrects for EBE shrinkage in the Ω M-step. Without it, the sample covariance of η̂ᵢ under-estimates Ω.
- **ITS Ω M-step**: `Ω_new = (1/N) Σᵢ [(η̂ᵢ − μ̂)(η̂ᵢ − μ̂)ᵀ + Ĉᵢ]` (closed form). The θ/σ M-step reuses `saem_theta_sigma_mstep` (BFGS on conditional obs-NLL with η fixed).
- **ITS convergence**: rolling window — compares mean of last `conv_window÷2` packed-param vectors against the previous `conv_window÷2`. Converged when max relative change < `rel_tol`.
- **Plotting**: examples use TidierPlots.jl (`ggplot`, `geom_point`, `geom_line`, `geom_hline`, `ggsave`). `geom_abline` is not implemented in TidierPlots — draw identity lines with `geom_line` + a two-point DataFrame. Use `:dash` (Symbol) not `"dashed"` (String) for `linetype`.
