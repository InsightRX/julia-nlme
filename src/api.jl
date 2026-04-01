"""
Public API for JuliaNLME.

Primary entry point:
  result = fit(model_path, data_path; kwargs...)
  result = fit(model, population, init_params; kwargs...)
"""

# ---------------------------------------------------------------------------
# Build ModelParameters from parsed specs
# ---------------------------------------------------------------------------

function _build_init_params(theta_specs, omega_specs, sigma_specs)
    theta_vals  = [s.initial for s in theta_specs]
    theta_names = [s.name    for s in theta_specs]

    # Assemble eta names in order
    eta_names = Symbol[]
    for spec in omega_specs, name in spec.names
        push!(eta_names, name)
    end
    n_eta = length(eta_names)

    # Build Ω matrix from specs
    omega_mat = zeros(n_eta, n_eta)
    eta_offset = 0
    for spec in omega_specs
        n = length(spec.names)
        if n == 1
            # Single diagonal element
            i = findfirst(==(spec.names[1]), eta_names)
            omega_mat[i, i] = spec.values[1]
        else
            # Lower-triangle block: values are [var1, cov12, var2, cov13, cov23, var3, ...]
            idxs = [findfirst(==(name), eta_names) for name in spec.names]
            k = 1
            for col in 1:n, row in col:n
                i, j = idxs[row], idxs[col]
                omega_mat[i, j] = spec.values[k]
                omega_mat[j, i] = spec.values[k]
                k += 1
            end
        end
    end

    omega = OmegaMatrix(omega_mat, eta_names)

    sigma_vals  = [s.value for s in sigma_specs]
    sigma_names = [s.name  for s in sigma_specs]
    sigma = SigmaMatrix(sigma_vals, sigma_names)

    return ModelParameters(theta_vals, theta_names, omega, sigma)
end

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
    fit(model, population; init_params=nothing, kwargs...)

Fit a compiled model to a population dataset.

Keyword arguments:
  - `init_params`:        `ModelParameters` with initial values. If `nothing`,
                          values are taken from the model file's [parameters] block.
  - `outer_maxiter`:      maximum outer iterations (default 500)
  - `outer_gtol`:         gradient tolerance for outer loop (default 1e-4)
  - `inner_maxiter`:      max inner iterations per subject (default 200)
  - `run_covariance_step`: compute SEs via Hessian inversion (default true)
  - `interaction`:        use FOCE-I (eta-epsilon interaction); evaluates R at IPRED
                          instead of PRED. Recommended for proportional/combined error
                          models. (default false = standard FOCE)
  - `verbose`:            print iteration progress (default true)
"""
function fit(model::CompiledModel,
             population::Population,
             init_params::ModelParameters;
             outer_maxiter::Int = 500,
             outer_gtol::Float64 = 1e-4,
             inner_maxiter::Int = 200,
             inner_tol::Float64 = 1e-6,
             run_covariance_step::Bool = true,
             interaction::Bool = false,
             verbose::Bool = true)::FitResult

    warnings = String[]

    # Validate covariates
    for subject in population.subjects
        for cov in _infer_required_covariates(model)
            if !haskey(subject.covariates, cov)
                push!(warnings, "Subject $(subject.id) missing covariate $cov — using 0.0")
                subject.covariates[cov] = 0.0
            end
        end
    end

    final_params, state, optim_result, covar, se_all =
        optimize_population(population, model, init_params;
                             outer_maxiter, outer_gtol,
                             inner_maxiter, inner_tol,
                             run_covariance_step, interaction, verbose)

    append!(warnings, state.inner_warnings)
    converged = Optim.converged(optim_result)

    if !converged
        push!(warnings, "Outer optimizer did not converge — results may be unreliable")
    end

    # Post-estimation
    sub_results = _compute_subject_results(population, model, final_params,
                                            state.eta_hats, state.H_mats; interaction)

    n_obs     = sum(s -> length(s.observations), population.subjects)
    n_params  = n_packed(final_params)
    ofv       = state.last_ofv
    aic       = ofv + 2 * n_params
    bic       = ofv + n_params * log(n_obs)

    # Split SE vector into theta/omega/sigma parts
    n_theta = length(final_params.theta)
    n_eta   = n_etas(final_params.omega)
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
        Optim.iterations(optim_result),
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
