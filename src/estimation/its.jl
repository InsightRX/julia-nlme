"""
ITS (Iterative Two-Stage) estimation for NLME population parameter estimation.

Reference: Bauer RJ (2019) NONMEM Tutorial Part II. CPT Pharmacometrics Syst Pharmacol.
           doi:10.1002/psp4.12422  (Section on ITS / deterministic two-stage)

Algorithm per iteration:
  E-step: MAP estimation of η̂ᵢ via find_ebe (same as inner loop of FOCE).
          Compute approximate conditional variance Ĉᵢ = [HᵢᵀVᵢ⁻¹Hᵢ + Ω⁻¹]⁻¹.
  M-step: Closed-form Ω update using bias-corrected sample covariance.
          BFGS on conditional obs-NLL for θ and σ (reuses saem_theta_sigma_mstep).

ITS is deterministic and typically fast but less accurate than FOCE for sparse data.
It is a good initializer for FOCE or a quick first estimate.

Convergence: rolling window — when the max absolute relative change of the
running parameter averages over the last `conv_window` iterations falls below
`rel_tol`, the algorithm is considered converged.
"""

# ---------------------------------------------------------------------------
# Ĉᵢ: approximate posterior covariance for subject i
# ---------------------------------------------------------------------------

"""
    compute_C_hat(H, R_diag, omega_chol)

Compute the approximate conditional variance matrix for one subject:
  Ĉ = [HᵀV⁻¹H + Ω⁻¹]⁻¹

where V = diag(R_diag) and Ω⁻¹ is computed from the Cholesky factor L of Ω
via  L⁻ᵀL⁻¹.

Returns an (n_eta × n_eta) symmetric positive-definite matrix.
"""
function compute_C_hat(H::Matrix{Float64},
                        R_diag::Vector{Float64},
                        omega_chol::LowerTriangular{Float64})
    # HᵀV⁻¹H — accumulate in n_eta × n_eta without constructing the full V⁻¹
    A = H' * Diagonal(1.0 ./ R_diag) * H          # n_eta × n_eta
    # Ω⁻¹ = (LLᵀ)⁻¹ = L⁻ᵀL⁻¹
    L_inv = inv(omega_chol)
    Omega_inv = L_inv' * L_inv                     # n_eta × n_eta
    # Regularize to guarantee positive definiteness despite numerical errors
    M = Symmetric(A .+ Omega_inv)
    C = try
        inv(cholesky(M))
    catch
        inv(cholesky(Symmetric(M + 1e-8 * I)))
    end
    return Matrix(C)
end

# ---------------------------------------------------------------------------
# Closed-form Ω M-step with Ĉᵢ bias correction
# ---------------------------------------------------------------------------

"""
    its_omega_mstep(eta_hats, C_hats, template)

Closed-form ITS M-step for Ω:
  μ̂   = (1/N) Σᵢ η̂ᵢ
  Ω_new = (1/N) Σᵢ [(η̂ᵢ − μ̂)(η̂ᵢ − μ̂)ᵀ + Ĉᵢ]

The Ĉᵢ term corrects for EBE shrinkage; without it this would be the
naive sample covariance, which under-estimates Ω for sparse data.

Returns an `OmegaMatrix` constructed from `Ω_new`.
"""
function its_omega_mstep(eta_hats::Vector{Vector{Float64}},
                          C_hats::Vector{Matrix{Float64}},
                          template::ModelParameters)
    N     = length(eta_hats)
    n_eta = length(eta_hats[1])
    mu    = sum(eta_hats) ./ N

    S = zeros(n_eta, n_eta)
    for i in 1:N
        d  = eta_hats[i] .- mu
        S .+= d * d' .+ C_hats[i]
    end
    S ./= N

    # Symmetrize to remove floating-point asymmetry
    S = Symmetric(S)
    return OmegaMatrix(Matrix(S), template.omega.eta_names;
                        diagonal = template.omega.diagonal)
end

# ---------------------------------------------------------------------------
# Convergence helper: rolling window on packed parameters
# ---------------------------------------------------------------------------

"""
    _its_converged(param_history, conv_window, rel_tol)

Return `true` when the max absolute relative change between the mean of the
last `conv_window ÷ 2` packed-parameter vectors and the mean of the previous
`conv_window ÷ 2` vectors is less than `rel_tol`.

Requires at least `conv_window + 1` entries in `param_history`.
"""
function _its_converged(param_history::Vector{Vector{Float64}},
                         conv_window::Int,
                         rel_tol::Float64)
    n = length(param_history)
    half = conv_window ÷ 2
    n < conv_window + 1 && return false

    recent = mean(param_history[n-half+1:n])
    prior  = mean(param_history[n-conv_window+1:n-half])

    max_rel = maximum(abs.((recent .- prior) ./ (abs.(prior) .+ 1e-10)))
    return max_rel < rel_tol
end

# ---------------------------------------------------------------------------
# Main ITS loop
# ---------------------------------------------------------------------------

"""
    run_its(population, model, init_params; kwargs...)

Execute the ITS (Iterative Two-Stage) algorithm.
Returns `(final_params, diagnostics)` where `diagnostics` is a NamedTuple.

Keyword arguments:
  `n_iter`          Maximum number of ITS iterations (default 100)
  `conv_window`     Rolling window length for convergence check (default 20)
  `rel_tol`         Relative tolerance for convergence (default 1e-4)
  `theta_maxiter`   BFGS iterations for the (θ,σ) M-step (default 30)
  `inner_maxiter`   Max inner (EBE) iterations per subject (default 200)
  `inner_tol`       EBE convergence tolerance (default 1e-8)
  `verbose`         Print progress (default true)
"""
function run_its(population::Population,
                  model::CompiledModel,
                  init_params::ModelParameters;
                  n_iter::Int          = 100,
                  conv_window::Int     = 20,
                  rel_tol::Float64     = 1e-4,
                  theta_maxiter::Int   = 30,
                  inner_maxiter::Int   = 200,
                  inner_tol::Float64   = 1e-8,
                  verbose::Bool        = true)

    N     = length(population)
    n_eta = model.n_eta

    theta_cur = copy(init_params.theta)
    omega_cur = copy(init_params.omega.matrix)
    sigma_cur = copy(init_params.sigma.values)

    param_history = Vector{Vector{Float64}}()
    converged     = false

    verbose && @info @sprintf(
        "ITS: %d subjects, %d ETAs, max %d iter",
        N, n_eta, n_iter)

    for k in 1:n_iter
        params_k = _rebuild_params(init_params, theta_cur, omega_cur, sigma_cur)

        # ---- E-step: MAP estimation + Jacobians ----
        eta_hats, H_mats, any_failed = run_inner_loop(population, params_k, model;
                                                        maxiter = inner_maxiter,
                                                        tol     = inner_tol)
        any_failed && verbose &&
            @warn @sprintf("ITS iter %d: some inner EBE optimizations failed", k)

        # ---- E-step: Ĉᵢ computation ----
        C_hats = Vector{Matrix{Float64}}(undef, N)
        Threads.@threads for i in 1:N
            ipred_i  = Float64.(compute_predictions(model, population.subjects[i],
                                                     theta_cur, eta_hats[i]))
            R_diag_i = compute_R_diag(model.error_model, ipred_i, sigma_cur)
            C_hats[i] = compute_C_hat(H_mats[i], R_diag_i, params_k.omega.chol)
        end

        # ---- M-step: closed-form Ω ----
        omega_new = its_omega_mstep(eta_hats, C_hats, init_params)

        # ---- M-step: BFGS for θ and σ ----
        theta_new, sigma_new = saem_theta_sigma_mstep(
            population, model, eta_hats,
            theta_cur, sigma_cur, init_params;
            maxiter = theta_maxiter)

        theta_cur = theta_new
        omega_cur = omega_new.matrix
        sigma_cur = sigma_new

        # Track packed params for convergence check
        params_packed = pack_params(_rebuild_params(init_params, theta_cur,
                                                     omega_cur, sigma_cur))
        push!(param_history, params_packed)

        # Check convergence
        if _its_converged(param_history, conv_window, rel_tol)
            converged = true
            verbose && @info @sprintf("ITS converged at iteration %d", k)
            break
        end

        if verbose && (k == 1 || k % 10 == 0)
            @info @sprintf("ITS iter %3d/%d", k, n_iter)
        end
    end

    !converged && verbose && @warn @sprintf(
        "ITS reached maximum iterations (%d) without convergence", n_iter)

    final_params = _rebuild_params(init_params, theta_cur, omega_cur, sigma_cur)
    diagnostics  = (converged = converged, n_iter = length(param_history),
                    param_history = param_history)
    return final_params, diagnostics
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    fit_its(model, population[, init_params]; kwargs...)

Fit a compiled model to population data using the ITS (Iterative Two-Stage) algorithm.

Returns a `FitResult` with the same structure as `fit()`.
The final OFV is evaluated via the FOCE/Laplace approximation so AIC/BIC
values are directly comparable with `fit()` and `fit_saem()` results.

ITS-specific kwargs:
  `n_iter`              Max ITS iterations,                   default 100
  `conv_window`         Rolling window for convergence check, default 20
  `rel_tol`             Relative parameter change tolerance,  default 1e-4
  `theta_maxiter`       BFGS iters for (θ,σ) M-step,         default 30
  `run_covariance_step` Compute SEs via Hessian,              default true
  `interaction`         FOCEI for final OFV,                  default false
  `verbose`             Print progress,                       default true
  `inner_maxiter`       Max inner (EBE) iterations,           default 200
  `inner_tol`           EBE convergence tolerance,            default 1e-8
"""
function fit_its(model::CompiledModel,
                  population::Population,
                  init_params::ModelParameters;
                  n_iter::Int              = 100,
                  conv_window::Int         = 20,
                  rel_tol::Float64         = 1e-4,
                  theta_maxiter::Int       = 30,
                  run_covariance_step::Bool = true,
                  interaction::Bool        = false,
                  verbose::Bool            = true,
                  inner_maxiter::Int       = 200,
                  inner_tol::Float64       = 1e-8)

    warnings = String[]
    for subject in population.subjects
        for cov in _infer_required_covariates(model)
            if !haskey(subject.covariates, cov)
                push!(warnings, "Subject $(subject.id) missing covariate $cov — using 0.0")
                subject.covariates[cov] = 0.0
            end
        end
    end

    t_start = time()
    final_params, diagnostics = run_its(population, model, init_params;
        n_iter, conv_window, rel_tol, theta_maxiter,
        inner_maxiter, inner_tol, verbose)
    t_elapsed = time() - t_start
    verbose && @info @sprintf("ITS completed in %.1f s", t_elapsed)

    # ---- Final EBEs and FOCE OFV ----
    verbose && @info "Computing final EBEs and FOCE OFV..."
    eta_hats, H_mats, _ = run_inner_loop(population, final_params, model;
                                          maxiter = inner_maxiter, tol = inner_tol)
    ofv_nll = foce_population_nll(final_params, population, model,
                                   eta_hats, H_mats; interaction)
    ofv = 2 * ofv_nll

    # ---- Covariance step ----
    covar, se_all, cov_success = if run_covariance_step
        verbose && @info "Running covariance step..."
        compute_covariance(pack_params(final_params), population, model,
                           final_params, eta_hats, H_mats; interaction)
    else
        n_full = n_packed(final_params)
        zeros(n_full, n_full), Float64[], true
    end

    run_covariance_step && !cov_success &&
        push!(warnings, "Covariance step failed — SEs not available")
    !diagnostics.converged &&
        push!(warnings, "ITS did not converge — results may be unreliable")

    # ---- Per-subject results ----
    sub_results = _compute_subject_results(population, model, final_params,
                                            eta_hats, H_mats; interaction)

    n_obs    = sum(s -> length(s.observations), population.subjects)
    n_params = n_packed(final_params)
    n_theta  = length(final_params.theta)
    n_eta    = n_etas(final_params.omega)
    n_chol   = final_params.omega.diagonal ? n_eta : n_eta * (n_eta + 1) ÷ 2

    aic = ofv + 2 * n_params
    bic = ofv + n_params * log(n_obs)

    se_theta = isempty(se_all) ? Float64[] : se_all[1:n_theta]
    se_omega = isempty(se_all) ? Float64[] : se_all[n_theta+1:n_theta+n_chol]
    se_sigma = isempty(se_all) ? Float64[] : se_all[n_theta+n_chol+1:end]

    verbose && @info @sprintf("ITS final OFV=%.3f  AIC=%.3f  BIC=%.3f",
                               ofv, aic, bic)

    return FitResult(
        model, diagnostics.converged,
        ofv, aic, bic,
        final_params.theta,
        final_params.theta_names,
        final_params.omega.matrix,
        final_params.sigma.values,
        covar, se_theta, se_omega, se_sigma,
        sub_results,
        n_obs, length(population), n_params, diagnostics.n_iter,
        interaction, warnings)
end

fit_its(model::CompiledModel, population::Population; kwargs...) =
    fit_its(model, population, model.default_params; kwargs...)
