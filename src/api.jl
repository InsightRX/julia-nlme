"""
Public API for JuliaNLME.

Primary entry point:
  result = fit(model_path, data_path; kwargs...)
  result = fit(model, population, init_params; kwargs...)
"""

import Random

# ---------------------------------------------------------------------------
# Post-estimation diagnostics
# ---------------------------------------------------------------------------

function _compute_subject_results(population::Population,
                                   model::CompiledModel,
                                   params::ModelParameters,
                                   eta_hats::Vector{Vector{Float64}},
                                   H_mats::Vector{Matrix{Float64}};
                                   interaction::Bool = false)

    results = SubjectResult[]
    for (i, subject) in enumerate(population.subjects)
        eta_hat = eta_hats[i]
        H       = H_mats[i]

        ipred = Float64.(compute_predictions(model, subject, params.theta, eta_hat))
        pred  = Float64.(compute_predictions(model, subject, params.theta, zeros(model.n_eta)))

        iw = iwres(subject.observations, ipred, model.error_model, params.sigma.values)
        cw = cwres(subject.observations, ipred, H, eta_hat,
                   model.error_model, params.sigma.values, params.omega.matrix)

        ofv_i = foce_subject_nll(eta_hat, H, subject, params, model; interaction)

        push!(results, SubjectResult(subject.id, eta_hat, ipred, pred,
                                      Float64.(iw), Float64.(cw), ofv_i))
    end
    return results
end

# ---------------------------------------------------------------------------
# Main fit() methods
# ---------------------------------------------------------------------------

"""
    fit(model_path, data_path; covariate_columns=nothing, kwargs...)

High-level convenience method: read model file and dataset from disk,
run FOCE estimation, return a `FitResult`.
"""
function fit(model_path::AbstractString,
             data_path::AbstractString;
             covariate_columns::Union{Nothing, Vector{Symbol}} = nothing,
             kwargs...)::FitResult

    model      = parse_model_file(model_path)
    population = read_data(data_path; covariate_columns)
    return fit(model, population; kwargs...)
end

"""
    fit(model, population; kwargs...)

Fit using the initial parameter values defined in the model file's `[parameters]` block.
Equivalent to `fit(model, population, model.default_params; kwargs...)`.
"""
function fit(model::CompiledModel, population::Population; kwargs...)::FitResult
    return fit(model, population, model.default_params; kwargs...)
end

"""
    fit(model, population, init_params; kwargs...)

Fit a compiled model to a population dataset.

Keyword arguments:
  - `outer_maxiter`:       maximum outer iterations (default 500)
  - `outer_gtol`:          gradient tolerance for outer loop (default 1e-6)
  - `inner_maxiter`:       max inner iterations per subject (default 200)
  - `inner_tol`:           inner EBE convergence tolerance (default 1e-8)
  - `run_covariance_step`: compute SEs via Hessian inversion (default true)
  - `interaction`:         use FOCE-I (eta-epsilon interaction); evaluates R at IPRED
                           instead of PRED. Recommended for proportional/combined error
                           models. (default false = standard FOCE)
  - `verbose`:             print iteration progress (default true)
  - `n_starts`:            number of optimization starts (default 1). When > 1, uses
                           Latin Hypercube Sampling across the parameter bounds to
                           generate space-filling starting points and returns the result
                           with the lowest OFV.
  - `rng`:                 random number generator used for LHS start generation.
  - `optimizer`:           local optimizer. `:lbfgs` (default), `:bfgs`, or any NLopt
                           gradient-based symbol (`:LD_SLSQP`, `:LD_MMA`, etc.).
  - `lbfgs_memory`:        L-BFGS history length (default 5). Smaller = more aggressive
                           Hessian resets. Only used when `optimizer=:lbfgs`.
  - `global_search`:       run a gradient-free global pre-search (NLopt GN_CRS2_LM)
                           before the local optimizer to identify the correct basin
                           (default false). Most useful for a single-start run.
  - `global_maxeval`:      max evaluations for the global phase (default 200 × n_params).
"""
function fit(model::CompiledModel,
             population::Population,
             init_params::ModelParameters;
             outer_maxiter::Int = 500,
             outer_gtol::Float64 = 1e-6,
             inner_maxiter::Int = 200,
             inner_tol::Float64 = 1e-8,
             run_covariance_step::Bool = true,
             interaction::Bool = false,
             verbose::Bool = true,
             n_starts::Int = 1,
             rng::Random.AbstractRNG = Random.default_rng(),
             optimizer::Symbol = :lbfgs,
             lbfgs_memory::Int = 5,
             global_search::Bool = false,
             global_maxeval::Int = 0)::FitResult

    # Validate covariates once (shared across all starts)
    warnings = String[]
    for subject in population.subjects
        for cov in _infer_required_covariates(model)
            if !haskey(subject.covariates, cov)
                push!(warnings, "Subject $(subject.id) missing covariate $cov — using 0.0")
                subject.covariates[cov] = 0.0
            end
        end
    end

    # ---------------------------------------------------------------------------
    # Multi-start: run n_starts optimizations, pick lowest OFV, then do one
    # final fit from those params (converges immediately) to get the covariance step.
    # ---------------------------------------------------------------------------
    if n_starts > 1
        starts    = _lhs_starts(init_params, n_starts, rng)
        best_params = init_params
        best_ofv    = Inf

        for (k, params_k) in enumerate(starts)
            verbose && @info @sprintf("Multi-start: run %d / %d", k, n_starts)
            try
                fp, st, _, _, _ = optimize_population(
                    population, model, params_k;
                    outer_maxiter, outer_gtol, inner_maxiter, inner_tol,
                    run_covariance_step = false, interaction, optimizer, lbfgs_memory,
                    global_search, global_maxeval,
                    verbose = false)
                ofv_k = 2 * st.best_ofv
                verbose && @info @sprintf("  → OFV = %.3f", ofv_k)
                if isfinite(ofv_k) && ofv_k < best_ofv
                    best_ofv    = ofv_k
                    best_params = fp
                end
            catch e
                verbose && @warn "Multi-start run $k failed: $e"
            end
        end

        verbose && @info @sprintf("Multi-start best OFV = %.3f — polishing...", best_ofv)
        # Recurse as single-start from best params (runs covariance step once).
        return fit(model, population, best_params;
                   outer_maxiter, outer_gtol, inner_maxiter, inner_tol,
                   run_covariance_step, interaction, verbose, optimizer, lbfgs_memory,
                   n_starts = 1)
    end

    # ---------------------------------------------------------------------------
    # Single-start path (also used as the final polishing step above)
    # ---------------------------------------------------------------------------
    final_params, state, optim_result, covar, se_all =
        optimize_population(population, model, init_params;
                             outer_maxiter, outer_gtol,
                             inner_maxiter, inner_tol,
                             run_covariance_step, interaction, verbose,
                             optimizer, lbfgs_memory,
                             global_search, global_maxeval)

    append!(warnings, state.inner_warnings)
    converged = optim_result.converged

    if !converged
        push!(warnings, "Outer optimizer did not converge — results may be unreliable")
    end

    # Post-estimation
    sub_results = _compute_subject_results(population, model, final_params,
                                            state.eta_hats, state.H_mats; interaction)

    n_obs    = sum(s -> length(s.observations), population.subjects)
    n_params = n_packed(final_params)
    n_theta  = length(final_params.theta)
    n_eta    = n_etas(final_params.omega)

    # OFV = 2 × NLL, matching NONMEM's convention.
    # NONMEM does not include the n_obs×log(2π) normalisation constant in its
    # reported OFV — it is a constant w.r.t. parameters and cancels in model
    # comparisons. Omitting it keeps OFV values directly comparable to NONMEM output.
    ofv = 2 * state.best_ofv
    aic = ofv + 2 * n_params
    bic = ofv + n_params * log(n_obs)
    n_chol  = final_params.omega.diagonal ? n_eta : n_eta * (n_eta + 1) ÷ 2
    n_sigma = length(final_params.sigma.values)

    se_theta = isempty(se_all) ? Float64[] : se_all[1:n_theta]
    se_omega = isempty(se_all) ? Float64[] : se_all[n_theta+1:n_theta+n_chol]
    se_sigma = isempty(se_all) ? Float64[] : se_all[n_theta+n_chol+1:end]

    return FitResult(
        model,
        converged,
        ofv, aic, bic,
        final_params.theta,
        final_params.theta_names,
        final_params.omega.matrix,
        final_params.sigma.values,
        covar, se_theta, se_omega, se_sigma,
        sub_results,
        n_obs,
        length(population),
        n_params,
        optim_result.iterations,
        interaction,
        warnings
    )
end

"""
    fit(model_string_or_path, population, init_params; kwargs...)

Convenience overload: parse a model string (if it contains newlines) or file path.
"""
function fit(model_src::AbstractString,
             population::Population,
             init_params::ModelParameters;
             kwargs...)::FitResult
    model = if occursin('\n', model_src)
        parse_model_string(model_src)
    else
        parse_model_file(model_src)
    end
    return fit(model, population, init_params; kwargs...)
end

# Placeholder — actual implementation inspects the compiled model's param function AST
# For now, return an empty list (covariate injection is handled with defaults)
_infer_required_covariates(::CompiledModel) = Symbol[]

# ---------------------------------------------------------------------------
# Multi-start helper
# ---------------------------------------------------------------------------

"""
    _lhs_starts(template, n_starts, rng)

Generate `n_starts` starting points using Latin Hypercube Sampling over the
optimizer's box bounds in packed (log-scale) space.

LHS divides each dimension into `n_starts` equal-probability intervals and
places exactly one sample per interval, with intervals shuffled independently
across dimensions. This guarantees uniform marginal coverage of the parameter
space — unlike Gaussian jitter, which clusters near the initial point — and
requires far fewer starts to achieve comparable exploration.

The first returned point is always `x0` (the user's initial params).
"""
function _lhs_starts(template::ModelParameters,
                      n_starts::Int,
                      rng::Random.AbstractRNG)
    x0, lower, upper = initial_packed(template)
    n = length(x0)

    starts = Vector{Vector{Float64}}(undef, n_starts)
    starts[1] = x0

    for k in 2:n_starts
        x_k = Vector{Float64}(undef, n)
        for j in 1:n
            # Divide [lower[j], upper[j]] into n_starts intervals;
            # sample uniformly from interval k (with random offset).
            lo_j = lower[j] + (k - 1) / n_starts * (upper[j] - lower[j])
            hi_j = lower[j] +  k      / n_starts * (upper[j] - lower[j])
            x_k[j] = lo_j + rand(rng) * (hi_j - lo_j)
        end
        starts[k] = x_k
    end

    # Shuffle the interval assignments independently across dimensions
    # so starts aren't all along the diagonal of the parameter space.
    for j in 1:n
        perm = Random.randperm(rng, n_starts - 1) .+ 1   # shuffle starts 2..n
        vals_j = [starts[k][j] for k in 2:n_starts]
        for (i, k) in enumerate(perm)
            starts[k][j] = vals_j[i]
        end
    end

    return [unpack_params(x, template) for x in starts]
end

# ---------------------------------------------------------------------------
# simulate() — forward simulation from a FitResult or given parameters
# ---------------------------------------------------------------------------

"""
    simulate(model, population, params; n_sim=1)

Simulate observations from the model for each subject in `population`.
Returns a DataFrame with columns: ID, TIME, IPRED, PRED, DV_SIM.
"""
function simulate(model::CompiledModel,
                   population::Population,
                   params::ModelParameters;
                   n_sim::Int = 1,
                   rng = Random.default_rng())

    rows = []
    for subject in population.subjects
        eta_hat = zeros(model.n_eta)  # population prediction
        ipred   = Float64.(compute_predictions(model, subject, params.theta, eta_hat))
        for j in eachindex(subject.obs_times)
            V    = residual_variance(model.error_model, ipred[j], params.sigma.values)
            dv_sim = ipred[j] + sqrt(V) * randn(rng)
            push!(rows, (ID=subject.id, TIME=subject.obs_times[j],
                         IPRED=ipred[j], DV_SIM=dv_sim))
        end
    end
    return DataFrame(rows)
end
