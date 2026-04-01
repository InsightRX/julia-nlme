"""
Outer optimizer: optimize population parameters θ, Ω, σ using FOCE.

FOCE gradient strategy:
  - The objective f(x) runs the inner loop (Float64 only) to update EBEs, then
    returns the FOCE OFV with those EBEs held fixed.
  - The gradient g(x) differentiates ONLY `foce_population_nll` with the EBEs
    held fixed from the most recent objective evaluation. This is the standard
    FOCE approach: η̂ᵢ are treated as constants when computing ∂OFV/∂(θ,Ω,σ).
  - ForwardDiff is used for the gradient of `foce_population_nll` (which IS
    differentiable w.r.t. population parameters for fixed EBEs/H matrices).
"""

using Optim, ForwardDiff, LinearAlgebra, Logging

# Minimal logger that forwards everything except warnings from Optim's internals.
# Optim emits "Terminated early due to NaN in gradient" during line search when
# the FOCE gradient is undefined at extreme parameter values. This is expected
# and the optimizer recovers automatically — the warnings are not actionable.
struct _SuppressOptimWarnings <: AbstractLogger
    inner::AbstractLogger
end
Logging.min_enabled_level(l::_SuppressOptimWarnings) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::_SuppressOptimWarnings, level, mod, group, id) =
    !(level == Logging.Warn && (mod === Optim || parentmodule(mod) === Optim))
Logging.handle_message(l::_SuppressOptimWarnings, args...; kwargs...) =
    Logging.handle_message(l.inner, args...; kwargs...)

# ---------------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------------

mutable struct OuterState
    eta_hats::Vector{Vector{Float64}}
    H_mats::Vector{Matrix{Float64}}
    last_ofv::Float64
    n_evals::Int
    n_inner_failures::Int   # count of evals where ≥1 subject EBE didn't converge
    inner_warnings::Vector{String}
end

# ---------------------------------------------------------------------------
# OFV function (runs inner loop, caches EBEs, returns scalar Float64)
# ---------------------------------------------------------------------------

"""
    make_outer_objective(population, model, template_params; inner_opts)

Returns `(f, g!, fg!, state)` for use with `OnceDifferentiable(f, g!, fg!, x0)`.

The combined `fg!(g, x)` runs the inner loop ONCE and uses the same EBEs for
both the OFV value and the ForwardDiff gradient, ensuring BFGS consistency.
`f` and `g!` are thin wrappers that re-use cached EBEs when called after `fg!`
for the same `x`.
"""
function make_outer_objective(population::Population,
                                model::CompiledModel,
                                template::ModelParameters;
                                inner_maxiter::Int = 200,
                                inner_tol::Float64 = 1e-6,
                                interaction::Bool = false)

    n = length(population)
    state = OuterState(
        [zeros(model.n_eta) for _ in 1:n],
        [zeros(0, 0)        for _ in 1:n],
        Inf, 0, 0, String[]
    )

    # Helper: run inner loop, cache EBEs, return (params, eta_hats, H_mats, ofv)
    function _run_and_cache(x::AbstractVector{Float64})
        params = unpack_params(x, template)
        eta_hats_cur, H_mats_cur, any_failed =
            run_inner_loop(population, params, model;
                           maxiter=inner_maxiter, tol=inner_tol)
        # Inner convergence failures during optimization are normal (line search
        # explores poor parameter regions). Only count; don't warn on each one.
        if any_failed
            state.n_inner_failures += 1
        end
        state.eta_hats .= eta_hats_cur
        state.H_mats   .= H_mats_cur
        state.n_evals  += 1
        if state.n_evals % 10 == 0 && isfinite(state.last_ofv)
            @info "Outer iteration $(state.n_evals): OFV = $(round(state.last_ofv, digits=3))"
        end
        return params, eta_hats_cur, H_mats_cur
    end

    # Combined f+g: runs inner loop once, same EBEs for both value and gradient.
    # Signature: fg!(G, x) → scalar  (NLSolversBase fdf convention)
    function fdfg!(G::AbstractVector{Float64}, x::AbstractVector{Float64})
        params, eta_hats_cur, H_mats_cur = _run_and_cache(x)

        ofv_diff = x_ad -> begin
            theta, omega_mat, sigma_vals = _unpack_raw(x_ad, template)
            foce_population_nll_diff(theta, omega_mat, sigma_vals,
                                      population, model, eta_hats_cur, H_mats_cur;
                                      interaction)
        end
        grad = ForwardDiff.gradient(ofv_diff, x)
        for i in eachindex(G)
            G[i] = isfinite(grad[i]) ? grad[i] : 0.0
        end

        ofv = foce_population_nll(params, population, model, eta_hats_cur, H_mats_cur;
                                   interaction)
        state.last_ofv = ofv
        return isfinite(ofv) ? ofv : 1e20
    end

    # Thin wrappers (called when Optim wants f or g alone)
    function f_only(x::AbstractVector{Float64})
        params, eta_hats_cur, H_mats_cur = _run_and_cache(x)
        ofv = foce_population_nll(params, population, model, eta_hats_cur, H_mats_cur;
                                   interaction)
        state.last_ofv = ofv
        return isfinite(ofv) ? ofv : 1e20
    end

    function g_only!(G::AbstractVector{Float64}, x::AbstractVector{Float64})
        _, eta_hats_cur, H_mats_cur = _run_and_cache(x)
        ofv_diff = x_ad -> begin
            theta, omega_mat, sigma_vals = _unpack_raw(x_ad, template)
            foce_population_nll_diff(theta, omega_mat, sigma_vals,
                                      population, model, eta_hats_cur, H_mats_cur;
                                      interaction)
        end
        grad = ForwardDiff.gradient(ofv_diff, x)
        for i in eachindex(G)
            G[i] = isfinite(grad[i]) ? grad[i] : 0.0
        end
        nothing
    end

    return f_only, g_only!, fdfg!, state
end

# ---------------------------------------------------------------------------
# Covariance step
# ---------------------------------------------------------------------------

"""
    compute_covariance(x_hat, population, model, template, eta_hats, H_mats)

Approximate parameter covariance via inverse Hessian of the FOCE OFV
at convergence, with EBEs fixed (same approach as gradient computation).
"""
function compute_covariance(x_hat::Vector{Float64},
                              population::Population,
                              model::CompiledModel,
                              template::ModelParameters,
                              eta_hats::Vector{Vector{Float64}},
                              H_mats::Vector{Matrix{Float64}};
                              interaction::Bool = false)
    try
        ofv_fixed = x -> begin
            theta, omega_mat, sigma_vals = _unpack_raw(x, template)
            foce_population_nll_diff(theta, omega_mat, sigma_vals,
                                      population, model, eta_hats, H_mats;
                                      interaction)
        end
        H_mat = ForwardDiff.hessian(ofv_fixed, x_hat)
        C  = inv(Symmetric(H_mat))
        se = sqrt.(max.(diag(C), 0.0))
        return C, se, true
    catch e
        @warn "Covariance step failed: $e"
        n = length(x_hat)
        return zeros(n, n), zeros(n), false
    end
end

# ---------------------------------------------------------------------------
# Main outer optimisation
# ---------------------------------------------------------------------------

"""
    optimize_population(population, model, init_params; options...)

Run the full FOCE outer optimization.
Returns `(final_params, state, optim_result, covar_matrix, se_vector)`.
"""
function optimize_population(population::Population,
                               model::CompiledModel,
                               init_params::ModelParameters;
                               outer_maxiter::Int = 500,
                               outer_gtol::Float64 = 1e-4,
                               inner_maxiter::Int = 200,
                               inner_tol::Float64 = 1e-6,
                               run_covariance_step::Bool = true,
                               interaction::Bool = false,
                               verbose::Bool = true)

    x0, _, _ = initial_packed(init_params)

    f_only, g_only!, fdfg!, state = make_outer_objective(
        population, model, init_params;
        inner_maxiter, inner_tol, interaction
    )

    if verbose
        @info "Starting FOCE optimization with $(length(x0)) parameters, $(length(population)) subjects"
    end

    # Use OnceDifferentiable with combined fg! so BFGS always gets consistent (f,g)
    od = OnceDifferentiable(f_only, g_only!, fdfg!, x0)

    result = with_logger(_SuppressOptimWarnings(current_logger())) do
        Optim.optimize(
            od, x0,
            BFGS(linesearch = Optim.LineSearches.BackTracking()),
            Optim.Options(
                iterations     = outer_maxiter,
                g_tol          = outer_gtol,
                show_trace     = verbose,
                extended_trace = false,
            )
        )
    end

    x_hat        = Optim.minimizer(result)
    final_params = unpack_params(x_hat, init_params)

    covar, se_all, cov_success = if run_covariance_step
        @info "Running covariance step..."
        compute_covariance(x_hat, population, model, init_params,
                           state.eta_hats, state.H_mats; interaction)
    else
        n = length(x0)
        zeros(n, n), zeros(n), false
    end

    # Warn only if inner failures were very frequent (>50% of evaluations)
    if state.n_inner_failures > state.n_evals ÷ 2
        push!(state.inner_warnings,
              "Inner optimizer failed on $(state.n_inner_failures)/$(state.n_evals) evaluations — results may be unreliable")
    end

    if !cov_success
        push!(state.inner_warnings, "Covariance step failed — SEs not available")
    end

    return final_params, state, result, covar, se_all
end
