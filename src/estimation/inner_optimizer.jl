"""
Inner optimizer: find Empirical Bayes Estimates (EBEs) for each subject.

For each subject i, minimizes:
  individual_nll(ηᵢ | θ, Ω, σ, data_i)

using Newton's method with ForwardDiff-computed gradients and Hessian.
Also computes the Jacobian H = ∂f/∂η|_{η̂} needed for FOCE linearization.
"""

using ForwardDiff, Optim, LinearAlgebra

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

    # Objective and gradient via ForwardDiff
    obj = η -> individual_nll(η, subject, params, model)

    grad! = (g, η) -> begin
        g .= ForwardDiff.gradient(obj, η)
        nothing
    end

    result = try
        Optim.optimize(
            obj, grad!, eta0,
            BFGS(linesearch = Optim.LineSearches.BackTracking()),
            Optim.Options(iterations = maxiter, g_tol = tol, show_trace = false)
        )
    catch
        # Fall back to gradient-free Nelder-Mead if BFGS fails
        try
            Optim.optimize(obj, eta0, NelderMead(),
                           Optim.Options(iterations = maxiter*10, g_tol = tol, show_trace = false))
        catch
            return eta0, zeros(length(subject.observations), n_eta), false
        end
    end

    eta_hat   = Optim.minimizer(result)
    converged = Optim.converged(result)

    # Jacobian H = ∂f/∂η at η̂  (n_obs × n_eta)
    H = ForwardDiff.jacobian(
        η -> compute_predictions(model, subject, params.theta, η),
        eta_hat
    )

    return eta_hat, H, converged
end

# ---------------------------------------------------------------------------
# Run inner loop over all subjects (parallelised)
# ---------------------------------------------------------------------------

"""
    run_inner_loop(population, params, model; kwargs...)

Find EBEs for all subjects. Returns `(eta_hats, H_mats, any_failed)`.

Subjects are processed in parallel using `Threads.@threads`.
"""
function run_inner_loop(population::Population,
                         params::ModelParameters,
                         model::CompiledModel;
                         maxiter::Int = 200,
                         tol::Float64 = 1e-6)

    n = length(population)
    eta_hats  = Vector{Vector{Float64}}(undef, n)
    H_mats    = Vector{Matrix{Float64}}(undef, n)
    converged = Vector{Bool}(undef, n)

    Threads.@threads for i in 1:n
        try
            eta_hats[i], H_mats[i], converged[i] =
                find_ebe(population[i], params, model; maxiter, tol)
        catch
            n_obs = length(population[i].observations)
            eta_hats[i]  = zeros(model.n_eta)
            H_mats[i]    = zeros(n_obs, model.n_eta)
            converged[i] = false
        end
    end

    any_failed = !all(converged)
    return eta_hats, H_mats, any_failed
end
