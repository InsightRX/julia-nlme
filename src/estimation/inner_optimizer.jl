"""
Inner optimizer: find Empirical Bayes Estimates (EBEs) for each subject.

For each subject i, minimizes:
  individual_nll(ηᵢ | θ, Ω, σ, data_i)

using Newton's method with ForwardDiff-computed gradients and Hessian.
Also computes the Jacobian H = ∂f/∂η|_{η̂} needed for FOCE linearization.
"""

using ForwardDiff, DiffResults, Optim, LinearAlgebra, Logging

# ---------------------------------------------------------------------------
# Find EBE for a single subject
# ---------------------------------------------------------------------------

"""
    find_ebe(subject, params, model; maxiter, tol)

Minimize `individual_nll` over η for one subject.

Returns `(eta_hat, H_matrix, converged)` where:
  - `eta_hat` is the vector of EBEs
  - `H_matrix` is the Jacobian ∂f/∂η evaluated at `eta_hat`  (n_obs × n_eta)
  - `converged` is a Bool

Uses BFGS with automatic differentiation. Starting point is η = 0.
"""
function find_ebe(subject::Subject,
                   params::ModelParameters,
                   model::CompiledModel;
                   maxiter::Int = 200,
                   tol::Float64 = 1e-6)

    n_eta = model.n_eta
    eta0  = zeros(n_eta)

    obj = η -> individual_nll(η, subject, params, model)

    # Pre-allocate DiffResult and GradientConfig once per find_ebe call.
    # Avoids reallocating Dual-number buffers on every BFGS gradient step.
    diff_result = DiffResults.GradientResult(eta0)
    cfg         = ForwardDiff.GradientConfig(obj, eta0)

    # Combined f+g: one Dual forward pass extracts both value and gradient.
    # OnceDifferentiable calls this at accepted BFGS steps, saving the extra
    # Float64 function evaluation that separate f/g! calls would require.
    function fg!(G, η)
        ForwardDiff.gradient!(diff_result, obj, η, cfg)
        G .= DiffResults.gradient(diff_result)
        return DiffResults.value(diff_result)
    end

    f_only(η) = individual_nll(η, subject, params, model)
    g_only!(G, η) = (fg!(G, η); nothing)

    # Suppress Optim's "NaN in gradient" warnings: they are expected when the
    # line search explores extreme η values and the individual NLL is undefined
    # there. The optimizer recovers automatically. NullLogger is safe here
    # because find_ebe itself emits no @info or @warn messages.
    # with_logger alone is not sufficient when called from Threads.@threads
    # tasks (which don't reliably inherit the parent task's logger state).
    #
    # NOTE: OnceDifferentiable construction is inside the try block.
    # For ODE models, the fg! call inside the constructor triggers
    # OrdinaryDiffEq JIT compilation for Dual types; when called from
    # multiple Threads.@threads workers simultaneously this can fail.
    # Keeping construction inside the try block ensures we fall back to
    # Nelder-Mead (Float64-only, no Dual compilation needed) in that case.
    result = try
        with_logger(NullLogger()) do
            od = OnceDifferentiable(f_only, g_only!, fg!, copy(eta0))
            Optim.optimize(
                od, eta0,
                BFGS(linesearch = Optim.LineSearches.BackTracking()),
                Optim.Options(iterations = maxiter, g_tol = tol, show_trace = false)
            )
        end
    catch
        # Fall back to gradient-free Nelder-Mead if BFGS fails
        try
            with_logger(NullLogger()) do
                Optim.optimize(f_only, eta0, NelderMead(),
                               Optim.Options(iterations = maxiter*10, g_tol = tol, show_trace = false))
            end
        catch
            return eta0, zeros(length(subject.observations), n_eta), false
        end
    end

    eta_hat   = Optim.minimizer(result)
    converged = Optim.converged(result)

    # Jacobian H = ∂f/∂η at η̂  (n_obs × n_eta).
    # Pre-allocate JacobianConfig to avoid re-allocating Dual buffers.
    pred_fn = η -> compute_predictions(model, subject, params.theta, η)
    jac_cfg = ForwardDiff.JacobianConfig(pred_fn, eta_hat)
    H = ForwardDiff.jacobian(pred_fn, eta_hat, jac_cfg)

    return eta_hat, H, converged
end

# ---------------------------------------------------------------------------
# Run inner loop over all subjects (parallelised)
# ---------------------------------------------------------------------------

"""
    run_inner_loop(population, params, model; kwargs...)

Find EBEs for all subjects. Returns `(eta_hats, H_mats, any_failed)`.

When `nthreads > 1`, subjects are processed in parallel using `Threads.@threads`.
"""
function run_inner_loop(population::Population,
                         params::ModelParameters,
                         model::CompiledModel;
                         maxiter::Int = 200,
                         tol::Float64 = 1e-6,
                         nthreads::Int = 1)

    n = length(population)
    eta_hats  = Vector{Vector{Float64}}(undef, n)
    H_mats    = Vector{Matrix{Float64}}(undef, n)
    converged = Vector{Bool}(undef, n)

    _ebe_body(i) = try
        eta_hats[i], H_mats[i], converged[i] =
            find_ebe(population[i], params, model; maxiter, tol)
    catch
        n_obs = length(population[i].observations)
        eta_hats[i]  = zeros(model.n_eta)
        H_mats[i]    = zeros(n_obs, model.n_eta)
        converged[i] = false
    end

    if nthreads > 1
        Threads.@threads for i in 1:n
            _ebe_body(i)
        end
    else
        for i in 1:n
            _ebe_body(i)
        end
    end

    any_failed = !all(converged)
    return eta_hats, H_mats, any_failed
end
