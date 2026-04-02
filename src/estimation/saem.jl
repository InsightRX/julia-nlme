"""
SAEM (Stochastic Approximation EM) for NLME population parameter estimation.

Reference: Delyon, Lavielle, Moulines (1999) Annals of Statistics 94–128.
           Kuhn & Lavielle (2004) ESAIM: Probability and Statistics 8:115–131.

Two-phase step-size schedule (Monolix convention):
  Phase 1 (exploration, k ≤ K1):  γₖ = 1          — rapid basin convergence
  Phase 2 (convergence, k > K1):  γₖ = 1/(k−K1)   — almost-sure convergence to MLE

Per-iteration:
  Simulation: MH sampling of ηᵢ from p(ηᵢ|yᵢ; θₖ₋₁, Ωₖ₋₁, σₖ₋₁)
  SA update:  s₂ ← (1−γ)s₂ + γ × (1/N)Σᵢ ηᵢηᵢᵀ
  M-step Ω:   Ωₖ = s₂                               (closed form)
  M-step θ,σ: argmin conditional obs-NLL             (BFGS, warm-started)

MH proposal: η_prop = η_curr + δᵢ × chol(Ω) × z, z∼N(0,I).
Step sizes δᵢ are adapted per subject every `adapt_interval` iterations
to target ~40% acceptance rate.

Final OFV uses the FOCE/Laplace approximation (same as `fit()`) so that
AIC/BIC values are directly comparable between SAEM and FOCE results.
"""

using Statistics: mean

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

saem_step(k::Int, k1::Int) = k <= k1 ? 1.0 : 1.0 / (k - k1)

# ---------------------------------------------------------------------------
# Metropolis-Hastings step for one subject
# ---------------------------------------------------------------------------

"""
    mh_steps!(eta, nll_current, subject, model, params, step_scale, rng, n_steps)

Run `n_steps` MH iterations for one subject in-place.

Proposal: η_prop = η + step_scale × L × z,  z ∼ N(0, I),  L = chol(Ω).

The log-target is  −individual_nll(η, ...)  (individual_nll already encodes
both the observation likelihood and the η prior, so the acceptance ratio is
just nll_current − nll_prop).

Returns `(n_accepted, updated_nll)`.
"""
function mh_steps!(eta::Vector{Float64},
                    nll_current::Float64,
                    subject::Subject,
                    model::CompiledModel,
                    params::ModelParameters,
                    step_scale::Float64,
                    rng::Random.AbstractRNG,
                    n_steps::Int)
    n_accepted = 0
    L = params.omega.chol
    d = length(eta)

    for _ in 1:n_steps
        eta_prop = eta .+ step_scale .* (L * randn(rng, d))
        nll_prop = individual_nll(eta_prop, subject, params, model)
        if log(rand(rng)) < nll_current - nll_prop   # log α = −nll_prop+nll_current
            eta        .= eta_prop
            nll_current = nll_prop
            n_accepted += 1
        end
    end
    return n_accepted, nll_current
end

# ---------------------------------------------------------------------------
# Conditional observation NLL  (θ, σ M-step objective)
# ---------------------------------------------------------------------------

"""
    conditional_obs_nll(theta, sigma_vals, population, model, eta_samples)

Sum of observation log-likelihoods with ETAs held fixed at their sampled values:
  Σᵢ Σⱼ [ ½ log Vᵢⱼ + ½(yᵢⱼ−fᵢⱼ)²/Vᵢⱼ ]

Generic over T for ForwardDiff auto-differentiation.
"""
function conditional_obs_nll(theta::AbstractVector{T},
                               sigma_vals::AbstractVector{T},
                               population::Population,
                               model::CompiledModel,
                               eta_samples::Vector{Vector{Float64}}) where T<:Real
    nll = zero(T)
    for (i, subject) in enumerate(population.subjects)
        ipred = compute_predictions(model, subject, theta, T.(eta_samples[i]))
        for (j, (y, f)) in enumerate(zip(subject.observations, ipred))
            f = max(f, T(1e-12))
            V = residual_variance(model.error_model, f, sigma_vals)
            V = max(V, T(1e-12))
            nll += T(0.5) * (log(V) + (T(y) - f)^2 / V)
        end
    end
    return nll
end

# ---------------------------------------------------------------------------
# M-step for (θ, σ): BFGS on conditional NLL with fixed ETAs
# ---------------------------------------------------------------------------

"""
    saem_theta_sigma_mstep(population, model, eta_samples, theta, sigma_vals,
                            template; maxiter)

Minimize `conditional_obs_nll` over (θ, σ) with ETAs held fixed.
Works in log-space; bounds come from the model file (θ) and conservative
defaults (σ).  Fixed parameters are pinned with ε-width bounds.

Returns `(new_theta, new_sigma_vals)` or the unchanged inputs on failure.
"""
function saem_theta_sigma_mstep(population::Population,
                                  model::CompiledModel,
                                  eta_samples::Vector{Vector{Float64}},
                                  theta::Vector{Float64},
                                  sigma_vals::Vector{Float64},
                                  template::ModelParameters;
                                  maxiter::Int = 30)
    n_theta = length(theta)
    n_sigma = length(sigma_vals)

    x0 = vcat(log.(theta), log.(sigma_vals))

    lower_th = log.(max.(template.theta_lower, 1e-10))
    upper_th = log.(min.(template.theta_upper, 1e9))
    lower = vcat(lower_th, fill(-8.0, n_sigma))
    upper = vcat(upper_th, fill( 5.0, n_sigma))

    # Pin fixed parameters (ε-width window; optimiser moves them by < 1e-10)
    _eps = 1e-10
    if !isempty(template.packed_fixed)
        n_eta  = n_etas(template.omega)
        n_chol = template.omega.diagonal ? n_eta : n_eta * (n_eta + 1) ÷ 2
        for i in 1:n_theta
            if i <= length(template.packed_fixed) && template.packed_fixed[i]
                lower[i] = x0[i] - _eps;  upper[i] = x0[i] + _eps
            end
        end
        for j in 1:n_sigma
            idx = n_theta + n_chol + j
            if idx <= length(template.packed_fixed) && template.packed_fixed[idx]
                lower[n_theta+j] = x0[n_theta+j] - _eps
                upper[n_theta+j] = x0[n_theta+j] + _eps
            end
        end
    end

    obj = x -> conditional_obs_nll(exp.(x[1:n_theta]), exp.(x[n_theta+1:end]),
                                    population, model, eta_samples)
    result = try
        od = OnceDifferentiable(obj, x0; autodiff = :forward)
        with_logger(NullLogger()) do
            Optim.optimize(od, lower, upper, x0,
                Fminbox(BFGS(linesearch = Optim.LineSearches.BackTracking())),
                Optim.Options(iterations = maxiter, g_tol = 1e-4, show_trace = false))
        end
    catch
        return theta, sigma_vals
    end

    x = Optim.minimizer(result)
    return exp.(x[1:n_theta]), exp.(x[n_theta+1:end])
end

# ---------------------------------------------------------------------------
# SAEM internal state
# ---------------------------------------------------------------------------

mutable struct SAEMState
    # SA sufficient statistic for Ω: running average of (1/N)Σᵢ ηᵢηᵢᵀ
    s2::Matrix{Float64}

    # Current parameter estimates
    theta::Vector{Float64}
    omega_mat::Matrix{Float64}
    sigma_vals::Vector{Float64}

    # Per-subject MH state
    eta_samples::Vector{Vector{Float64}}
    nll_cache::Vector{Float64}       # cached NLL at current η (avoids re-evaluation)
    step_scales::Vector{Float64}     # per-subject MH step sizes δᵢ
    accept_counts::Vector{Int}       # accepts since last adaptation
    steps_since_adapt::Int

    # Diagnostics
    cond_nll_trace::Vector{Float64}  # conditional obs-NLL per iteration (monitoring)
    inner_warnings::Vector{String}
end

# ---------------------------------------------------------------------------
# Main SAEM loop
# ---------------------------------------------------------------------------

"""
    run_saem(population, model, init_params; kwargs...)

Execute the two-phase SAEM algorithm. Returns `(final_params, state)`.

Keyword arguments
  `n_iter_exploration`  Phase-1 iterations (γ=1), default 150
  `n_iter_convergence`  Phase-2 iterations (γ=1/k), default 250
  `n_mh_steps`          MH steps per subject per iteration, default 2
  `adapt_interval`      Iterations between step-scale updates, default 50
  `target_accept`       Target MH acceptance rate (per step), default 0.4
  `saem_theta_maxiter`  BFGS iterations for the (θ,σ) M-step, default 30
  `verbose`             Print iteration progress, default true
  `rng`                 Master RNG (spawns per-thread children for safety)
"""
function run_saem(population::Population,
                   model::CompiledModel,
                   init_params::ModelParameters;
                   n_iter_exploration::Int  = 150,
                   n_iter_convergence::Int  = 250,
                   n_mh_steps::Int          = 2,
                   adapt_interval::Int      = 50,
                   target_accept::Float64   = 0.4,
                   saem_theta_maxiter::Int  = 30,
                   verbose::Bool            = true,
                   rng::Random.AbstractRNG  = Random.default_rng())

    N      = length(population)
    n_eta  = model.n_eta
    n_iter = n_iter_exploration + n_iter_convergence

    # Per-thread RNGs (seeded from master) for thread-safe MH
    thread_rngs = [Random.MersenneTwister(rand(rng, UInt32))
                   for _ in 1:Threads.nthreads()]

    # Initialise current estimates
    theta_cur  = copy(init_params.theta)
    omega_cur  = copy(init_params.omega.matrix)
    sigma_cur  = copy(init_params.sigma.values)
    s2         = copy(omega_cur)   # SA stat starts at Ω₀ (consistent with η=0)

    eta_samples = [zeros(n_eta) for _ in 1:N]
    step_scales = fill(0.3, N)
    nll_cache   = zeros(N)

    # Initial NLL at η = 0
    params0 = _rebuild_params(init_params, theta_cur, omega_cur, sigma_cur)
    for (i, subject) in enumerate(population.subjects)
        nll_cache[i] = individual_nll(eta_samples[i], subject, params0, model)
    end

    state = SAEMState(s2, theta_cur, omega_cur, sigma_cur,
                       eta_samples, nll_cache, step_scales,
                       zeros(Int, N), 0, Float64[], String[])

    verbose && @info @sprintf(
        "SAEM: %d subjects, %d ETAs, %d total iter (%d explore + %d converge)",
        N, n_eta, n_iter, n_iter_exploration, n_iter_convergence)

    for k in 1:n_iter
        γ = saem_step(k, n_iter_exploration)

        # Rebuild ModelParameters with current estimates (cheap, just wraps arrays)
        params_k = _rebuild_params(init_params, state.theta, state.omega_mat,
                                    state.sigma_vals)

        # ---- Step 1: MH simulation ----
        Threads.@threads for i in 1:N
            tid = Threads.threadid()
            n_acc, nll_new = mh_steps!(
                state.eta_samples[i], state.nll_cache[i],
                population.subjects[i], model, params_k,
                state.step_scales[i], thread_rngs[tid], n_mh_steps)
            state.accept_counts[i] += n_acc
            state.nll_cache[i]      = nll_new
        end
        state.steps_since_adapt += 1

        # ---- Step 2: SA update of sufficient statistic for Ω ----
        eta_outer = zeros(n_eta, n_eta)
        for i in 1:N
            eta_outer .+= state.eta_samples[i] * state.eta_samples[i]'
        end
        eta_outer ./= N

        @. state.s2 = (1 - γ) * state.s2 + γ * eta_outer

        # ---- Step 3: M-step Ω (closed form) ----
        omega_new = OmegaMatrix(state.s2, init_params.omega.eta_names;
                                 diagonal = init_params.omega.diagonal)
        state.omega_mat .= omega_new.matrix

        # ---- Step 4: M-step (θ, σ) via BFGS on conditional NLL ----
        theta_new, sigma_new = saem_theta_sigma_mstep(
            population, model, state.eta_samples,
            state.theta, state.sigma_vals, init_params;
            maxiter = saem_theta_maxiter)
        state.theta      .= theta_new
        state.sigma_vals .= sigma_new

        # Update NLL cache since params changed
        params_upd = _rebuild_params(init_params, state.theta, state.omega_mat,
                                      state.sigma_vals)
        for i in 1:N
            state.nll_cache[i] = individual_nll(state.eta_samples[i],
                                                  population.subjects[i],
                                                  params_upd, model)
        end

        # ---- Adapt MH step sizes ----
        if state.steps_since_adapt >= adapt_interval
            for i in 1:N
                rate = state.accept_counts[i] / (n_mh_steps * adapt_interval)
                state.step_scales[i] = rate > target_accept ?
                    min(state.step_scales[i] * 1.1, 5.0) :
                    max(state.step_scales[i] * 0.9, 0.01)
                state.accept_counts[i] = 0
            end
            state.steps_since_adapt = 0
        end

        # ---- Track conditional NLL ----
        cond_nll = conditional_obs_nll(state.theta, state.sigma_vals,
                                        population, model, state.eta_samples)
        push!(state.cond_nll_trace, cond_nll)

        if verbose && (k == 1 || k % 50 == 0 || k == n_iter)
            phase = k <= n_iter_exploration ? "explore" : "converge"
            @info @sprintf("SAEM iter %4d/%d [%s] γ=%.3f  condNLL=%.3f",
                            k, n_iter, phase, γ, cond_nll)
        end
    end

    omega_fin  = OmegaMatrix(state.omega_mat, init_params.omega.eta_names;
                              diagonal = init_params.omega.diagonal)
    final_params = ModelParameters(state.theta, init_params.theta_names,
                                    init_params.theta_lower, init_params.theta_upper,
                                    omega_fin,
                                    SigmaMatrix(state.sigma_vals, init_params.sigma.names),
                                    init_params.packed_fixed)
    return final_params, state
end

# Helper: rebuild ModelParameters around updated arrays without re-allocating names/bounds.
function _rebuild_params(template::ModelParameters,
                          theta::Vector{Float64},
                          omega_mat::Matrix{Float64},
                          sigma_vals::Vector{Float64})
    omega = OmegaMatrix(omega_mat, template.omega.eta_names;
                         diagonal = template.omega.diagonal)
    ModelParameters(theta, template.theta_names,
                    template.theta_lower, template.theta_upper,
                    omega,
                    SigmaMatrix(sigma_vals, template.sigma.names),
                    template.packed_fixed)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    fit_saem(model, population[, init_params]; kwargs...)

Fit a compiled model to population data using the SAEM algorithm.

Returns a `FitResult` with the same structure as `fit()`.
The final OFV is evaluated via the FOCE/Laplace approximation so AIC/BIC
values are directly comparable with `fit()` results.

SAEM-specific kwargs (all others forwarded to covariance/EBE steps):
  `n_iter_exploration`  Phase-1 iterations (γ=1),        default 150
  `n_iter_convergence`  Phase-2 iterations (γ=1/(k−K₁)), default 250
  `n_mh_steps`          MH steps per subject per iter,   default 2
  `adapt_interval`      Iter between MH step-size adapts, default 50
  `saem_theta_maxiter`  BFGS iters for (θ,σ) M-step,    default 30
  `run_covariance_step` Compute SEs via Hessian,          default true
  `interaction`         FOCEI for final OFV,              default false
  `verbose`             Print progress,                   default true
  `rng`                 Random number generator
"""
function fit_saem(model::CompiledModel,
                   population::Population,
                   init_params::ModelParameters;
                   n_iter_exploration::Int   = 150,
                   n_iter_convergence::Int   = 250,
                   n_mh_steps::Int           = 2,
                   adapt_interval::Int       = 50,
                   saem_theta_maxiter::Int   = 30,
                   run_covariance_step::Bool = true,
                   interaction::Bool         = false,
                   verbose::Bool             = true,
                   rng::Random.AbstractRNG   = Random.default_rng(),
                   inner_maxiter::Int        = 200,
                   inner_tol::Float64        = 1e-8)

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
    final_params, saem_state = run_saem(population, model, init_params;
        n_iter_exploration, n_iter_convergence,
        n_mh_steps, adapt_interval, saem_theta_maxiter,
        verbose, rng)
    t_elapsed = time() - t_start
    verbose && @info @sprintf("SAEM completed in %.1f s", t_elapsed)

    # ---- Final EBEs and FOCE OFV (for AIC/BIC comparability with FOCE) ----
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
    append!(warnings, saem_state.inner_warnings)

    # ---- Per-subject results ----
    sub_results = _compute_subject_results(population, model, final_params,
                                            eta_hats, H_mats; interaction)

    n_obs    = sum(s -> length(s.observations), population.subjects)
    n_params = n_packed(final_params)
    n_theta  = length(final_params.theta)
    n_eta    = n_etas(final_params.omega)
    n_chol   = final_params.omega.diagonal ? n_eta : n_eta * (n_eta + 1) ÷ 2
    n_sigma  = length(final_params.sigma.values)
    aic      = ofv + 2 * n_params
    bic      = ofv + n_params * log(n_obs)

    se_theta = isempty(se_all) ? Float64[] : se_all[1:n_theta]
    se_omega = isempty(se_all) ? Float64[] : se_all[n_theta+1:n_theta+n_chol]
    se_sigma = isempty(se_all) ? Float64[] : se_all[n_theta+n_chol+1:end]

    verbose && @info @sprintf("SAEM final OFV=%.3f  AIC=%.3f  BIC=%.3f",
                               ofv, aic, bic)

    n_iter_total = n_iter_exploration + n_iter_convergence
    return FitResult(
        model, true,
        ofv, aic, bic,
        final_params.theta,
        final_params.theta_names,
        final_params.omega.matrix,
        final_params.sigma.values,
        covar, se_theta, se_omega, se_sigma,
        sub_results,
        n_obs, length(population), n_params, n_iter_total,
        interaction, warnings)
end

fit_saem(model::CompiledModel, population::Population; kwargs...) =
    fit_saem(model, population, model.default_params; kwargs...)
