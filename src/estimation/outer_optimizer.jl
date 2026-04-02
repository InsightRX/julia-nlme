"""
Outer optimizer: optimize population parameters θ, Ω, σ using FOCE/FOCEI.

FOCE gradient strategy:
  - The objective f(x) runs the inner loop (Float64 only) to update EBEs, then
    returns the FOCE/FOCEI OFV with those EBEs held fixed.
  - The gradient g(x) differentiates ONLY `foce_population_nll` with the EBEs
    held fixed from the most recent objective evaluation. This is the standard
    FOCE approach: η̂ᵢ are treated as constants when computing ∂OFV/∂(θ,Ω,σ).
  - ForwardDiff is used for the gradient of `foce_population_nll` (which IS
    differentiable w.r.t. population parameters for fixed EBEs/H matrices).
"""

using Optim, NLopt, ForwardDiff, LinearAlgebra, Logging, Printf

# ---------------------------------------------------------------------------
# Unified result type — abstracts over Optim.jl and NLopt return values
# so the rest of the code doesn't need to know which backend was used.
# ---------------------------------------------------------------------------
struct _OptResult
    minimizer::Vector{Float64}
    converged::Bool
    iterations::Int
    g_residual::Float64   # ‖∇f‖∞ at solution
end

# Construct from an Optim result
_OptResult(r::Optim.OptimizationResults) = _OptResult(
    Optim.minimizer(r),
    Optim.converged(r),
    Optim.iterations(r),
    Optim.g_residual(r),
)

# Logger that suppresses all Warn-level messages within the Optim.optimize block.
# All @warn calls in that block come from Optim / LineSearches internals (e.g.
# "Terminated early due to NaN in gradient") — they are expected during line
# search when the FOCE gradient is undefined at extreme parameter values, and
# the optimizer recovers automatically.  Our own code inside that block uses
# only @info for progress; no legitimate warnings are lost.
# Suppressing by level (not by module) avoids === identity and string-matching
# fragilities that caused earlier approaches to silently fail.
struct _SuppressOptimWarnings <: AbstractLogger
    inner::AbstractLogger
end
Logging.min_enabled_level(l::_SuppressOptimWarnings) = Logging.min_enabled_level(l.inner)
Logging.shouldlog(l::_SuppressOptimWarnings, level, mod, group, id) =
    level !== Logging.Warn
Logging.handle_message(l::_SuppressOptimWarnings, args...; kwargs...) =
    Logging.handle_message(l.inner, args...; kwargs...)

# ---------------------------------------------------------------------------
# State cache
# ---------------------------------------------------------------------------

mutable struct OuterState
    eta_hats::Vector{Vector{Float64}}
    H_mats::Vector{Matrix{Float64}}
    last_ofv::Float64   # OFV of the most recently evaluated point (may be a rejected trial)
    best_ofv::Float64   # best (lowest) OFV seen so far across all accepted points
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
                                interaction::Bool = false,
                                verbose::Bool = true)

    n = length(population)

    state = OuterState(
        [zeros(model.n_eta) for _ in 1:n],
        [zeros(0, 0)        for _ in 1:n],
        Inf, Inf, 0, 0, String[]
    )

    # Helper: run inner loop, cache EBEs, return (params, eta_hats, H_mats)
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
        return params, eta_hats_cur, H_mats_cur
    end

    # Log helper: called after best_ofv is updated so the message is always accurate.
    # OFV = 2×NLL, matching NONMEM's convention (no n_obs×log(2π) constant).
    function _log_improvement(eval_idx::Int)
        verbose || return
        @info @sprintf("Evaluation %d: OFV = %.3f", eval_idx, 2 * state.best_ofv)
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
        if isfinite(ofv) && ofv < state.best_ofv
            state.best_ofv = ofv
            _log_improvement(state.n_evals)
        end
        return isfinite(ofv) ? ofv : 1e20
    end

    # Thin wrappers (called when Optim wants f or g alone)
    function f_only(x::AbstractVector{Float64})
        params, eta_hats_cur, H_mats_cur = _run_and_cache(x)
        ofv = foce_population_nll(params, population, model, eta_hats_cur, H_mats_cur;
                                   interaction)
        state.last_ofv = ofv
        if isfinite(ofv) && ofv < state.best_ofv
            state.best_ofv = ofv
            _log_improvement(state.n_evals)
        end
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
    compute_covariance(x_hat, population, model, template, eta_hats, H_mats; interaction)

Approximate parameter covariance via inverse Hessian of the FOCE/FOCEI OFV
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

Run the full FOCE/FOCEI outer optimization.
Returns `(final_params, state, optim_result, covar_matrix, se_vector)`.
"""
function optimize_population(population::Population,
                               model::CompiledModel,
                               init_params::ModelParameters;
                               outer_maxiter::Int = 500,
                               outer_gtol::Float64 = 1e-6,
                               inner_maxiter::Int = 200,
                               inner_tol::Float64 = 1e-8,
                               run_covariance_step::Bool = true,
                               interaction::Bool = false,
                               verbose::Bool = true,
                               optimizer::Symbol = :lbfgs,
                               lbfgs_memory::Int = 5,
                               global_search::Bool = false,
                               global_maxeval::Int = 0)   # 0 → auto: 200 × n_params

    x0_full, lower_full, upper_full = initial_packed(init_params)

    f_only, g_only!, fdfg!, state = make_outer_objective(
        population, model, init_params;
        inner_maxiter, inner_tol, interaction, verbose
    )

    # -------------------------------------------------------------------------
    # Exclude fixed parameters from the optimizer's search space.
    # Fixed params have packed_fixed[i] = true; we pass only free dims to the
    # optimizer. Inside each objective call we expand the free vector back to
    # the full vector (with fixed values held at x0) before evaluation.
    # This avoids the 1/ε² Fminbox log-barrier curvature blow-up that would
    # otherwise dominate the Hessian and stall free-parameter updates.
    # -------------------------------------------------------------------------
    free_idx = isempty(init_params.packed_fixed) ?
        collect(1:length(x0_full)) :
        findall(.!init_params.packed_fixed)

    x0_fixed = x0_full   # full vector; fixed values never change
    x0    = x0_full[free_idx]
    lower = lower_full[free_idx]
    upper = upper_full[free_idx]

    # Expand a free sub-vector back to the full packed vector.
    expand(x_free) = (v = copy(x0_fixed); v[free_idx] = x_free; v)

    # Wrap the three objective closures to operate on the free sub-vector.
    f_free(x_free) = f_only(expand(x_free))

    function g_free!(G_free, x_free)
        G_full = zeros(length(x0_fixed))
        g_only!(G_full, expand(x_free))
        G_free .= G_full[free_idx]
        nothing
    end

    function fg_free!(G_free, x_free)
        G_full = zeros(length(x0_fixed))
        val = fdfg!(G_full, expand(x_free))
        G_free .= G_full[free_idx]
        val
    end

    # -------------------------------------------------------------------------
    # Optional global pre-search via NLopt GN_CRS2_LM (gradient-free).
    # CRS2_LM (Controlled Random Search + Local Mutation) maintains a population
    # of points across the full box and applies random + local perturbations.
    # It identifies the basin; the local phase then polishes to the minimum.
    # -------------------------------------------------------------------------
    if global_search
        n_free = length(x0)
        max_ev = global_maxeval > 0 ? global_maxeval : 200 * n_free
        verbose && @info @sprintf("Global pre-search (GN_CRS2_LM, max %d evals)...", max_ev)
        opt_g = NLopt.Opt(:GN_CRS2_LM, n_free)
        NLopt.lower_bounds!(opt_g, lower)
        NLopt.upper_bounds!(opt_g, upper)
        NLopt.maxeval!(opt_g, max_ev)
        NLopt.min_objective!(opt_g, (x, _) -> f_free(x))
        (_, x_global, _) = NLopt.optimize(opt_g, x0)
        verbose && @info @sprintf("  Global phase best OFV = %.3f — starting local polish",
                                   2 * state.best_ofv)
        x0 = x_global   # hand off basin to local optimizer
    end

    t_start = time()
    opt_result = if optimizer === :bfgs || optimizer === :lbfgs
        # -----------------------------------------------------------------
        # Optim.jl path (BFGS / L-BFGS via Fminbox)
        # -----------------------------------------------------------------
        od = OnceDifferentiable(f_free, g_free!, fg_free!, x0)
        inner_opt = optimizer === :bfgs ?
            BFGS(linesearch = Optim.LineSearches.BackTracking()) :
            LBFGS(m = lbfgs_memory, linesearch = Optim.LineSearches.BackTracking())

        raw = with_logger(_SuppressOptimWarnings(current_logger())) do
            Optim.optimize(
                od, lower, upper, x0,
                Fminbox(inner_opt),
                Optim.Options(
                    iterations     = outer_maxiter,
                    g_tol          = outer_gtol,
                    show_trace     = false,
                    extended_trace = false,
                )
            )
        end
        _OptResult(raw)
    else
        # -----------------------------------------------------------------
        # NLopt path — any NLopt.jl gradient-based algorithm symbol,
        # e.g. :LD_LBFGS, :LD_SLSQP, :LD_MMA, :LD_TNEWTON_PRECOND_RESTART
        # -----------------------------------------------------------------
        n_free = length(x0)
        nlopt_algo = try
            NLopt.Algorithm(optimizer)
        catch
            error("Unknown optimizer :$optimizer. Use :bfgs, :lbfgs, or a NLopt " *
                  "gradient-based algorithm symbol such as :LD_LBFGS or :LD_SLSQP.")
        end

        opt = NLopt.Opt(nlopt_algo, n_free)
        NLopt.lower_bounds!(opt, lower)
        NLopt.upper_bounds!(opt, upper)
        NLopt.maxeval!(opt, outer_maxiter * (n_free + 1))  # NLopt counts per f/g call
        NLopt.xtol_rel!(opt, outer_gtol)                   # stop when ‖Δx‖/‖x‖ < gtol

        n_evals_nlopt = Ref(0)

        NLopt.min_objective!(opt, (x, grad) -> begin
            n_evals_nlopt[] += 1
            if length(grad) > 0
                return fg_free!(grad, x)
            else
                return f_free(x)
            end
        end)

        (minf, minx, ret) = NLopt.optimize(opt, x0)

        # Compute gradient at solution for the g_residual field
        g_final = zeros(n_free)
        g_free!(g_final, minx)
        g_norm = maximum(abs, g_final)

        converged_nlopt = ret in (:SUCCESS, :STOPVAL_REACHED,
                                   :FTOL_REACHED, :XTOL_REACHED)
        _OptResult(minx, converged_nlopt, n_evals_nlopt[], g_norm)
    end
    t_elapsed = time() - t_start

    # Expand the optimizer's free-parameter solution back to the full vector.
    x_hat_free   = opt_result.minimizer
    x_hat        = expand(x_hat_free)
    final_params = unpack_params(x_hat, init_params)

    if verbose
        status = opt_result.converged ? "converged" : "max iterations reached"
        @info @sprintf("FOCE optimization %s in %d evaluations (%.1f s, |∇|∞ = %.2e)",
                       status, opt_result.iterations, t_elapsed, opt_result.g_residual)
    end

    # Final inner loop to ensure state is consistent with the optimal parameters.
    eta_final, H_final, _ = run_inner_loop(population, final_params, model;
                                            maxiter=inner_maxiter, tol=inner_tol)
    state.eta_hats .= eta_final
    state.H_mats   .= H_final

    # Recompute best_ofv at the final parameters.
    # OFV = 2×NLL, matching NONMEM's convention (no n_obs×log(2π) constant).
    ofv_nll = foce_population_nll(final_params, population, model,
                                   eta_final, H_final; interaction)
    state.best_ofv = ofv_nll

    if verbose
        @info @sprintf("Final OFV = %.3f", 2 * ofv_nll)
    end

    covar, se_all, cov_success = if run_covariance_step
        verbose && @info "Running covariance step..."
        x_final = pack_params(final_params)
        compute_covariance(x_final, population, model, final_params,
                           eta_final, H_final; interaction)
    else
        n_full = length(x0_fixed)
        zeros(n_full, n_full), Float64[], true   # empty SE = skipped (not failed)
    end

    # Warn only if inner failures were very frequent (>50% of evaluations)
    if state.n_inner_failures > state.n_evals ÷ 2
        push!(state.inner_warnings,
              "Inner optimizer failed on $(state.n_inner_failures)/$(state.n_evals) evaluations — results may be unreliable")
    end

    if run_covariance_step && !cov_success
        push!(state.inner_warnings, "Covariance step failed — SEs not available")
    end

    return final_params, state, opt_result, covar, se_all
end
