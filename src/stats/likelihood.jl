"""
Likelihood functions for NLME estimation.

The individual log-likelihood (Laplacian approximation) is used in the inner
η-optimization. The FOCE approximation of the population objective function
is used in the outer optimization over θ, Ω, σ.
"""

using LinearAlgebra, LogExpFunctions

# ---------------------------------------------------------------------------
# Individual predictions given (theta, eta)
# ---------------------------------------------------------------------------

"""
    compute_predictions(model, subject, theta, eta)

Compute individual predicted concentrations at all observation times.
Returns Vector{T} (T may be a ForwardDiff Dual number).
"""
function compute_predictions(model::CompiledModel,
                               subject::Subject,
                               theta::AbstractVector{T1},
                               eta::AbstractVector{T2}) where {T1<:Real, T2<:Real}
    T = promote_type(T1, T2)

    if isempty(subject.tvcov)
        # Time-constant covariates: original single-pass path
        cov_T = Dict{Symbol, T}(k => T(v) for (k, v) in subject.covariates)
        pk_params = model.pk_param_fn(T.(theta), T.(eta), cov_T)
        return predict_subject(model.pk_model, pk_params, subject)
    else
        # Time-varying covariates: evaluate PK params separately at each obs time,
        # using the LOCF covariate value at that time, then run full superposition.
        n     = length(subject.obs_times)
        preds = Vector{T}(undef, n)
        # Base dict from time-constant covariates (merged with TV values below)
        base_cov = Dict{Symbol, T}(k => T(v) for (k, v) in subject.covariates)
        for j in 1:n
            cov_j = copy(base_cov)
            for (k, vals) in subject.tvcov
                cov_j[k] = T(vals[j])
            end
            pk_params_j   = model.pk_param_fn(T.(theta), T.(eta), cov_j)
            single_dose_fn = make_single_dose_fn(model.pk_model, pk_params_j)
            preds[j] = predict_concentration(single_dose_fn, subject.doses,
                                              T(subject.obs_times[j]))
        end
        return preds
    end
end

# ---------------------------------------------------------------------------
# Individual log-likelihood (Laplacian approximation objective)
# ---------------------------------------------------------------------------

"""
    individual_nll(eta, subject, params, model)

Negative log-likelihood for one subject at fixed η.
Used as the objective for the inner η-optimization.

NLL = 0.5 * [ ηᵀΩ⁻¹η + log|Ω| + Σⱼ((yⱼ-fⱼ)²/Vⱼ + log Vⱼ) ]

where Vⱼ = residual_variance(error_model, fⱼ, σ).

This is the exact Laplacian (not FOCE approximation) — used only for
finding the EBEs in the inner loop.
"""
function individual_nll(eta::AbstractVector{T},
                         subject::Subject,
                         params::ModelParameters,
                         model::CompiledModel) where T<:Real
    theta = T.(params.theta)
    ipred = compute_predictions(model, subject, theta, eta)

    n   = length(subject.observations)
    nll = zero(T)

    # Residual contribution: Σⱼ [ (y - f)²/V + log V ]
    for j in 1:n
        V    = residual_variance(model.error_model, ipred[j], params.sigma.values)
        resj = T(subject.observations[j]) - ipred[j]
        nll += resj^2 / V + log(V)
    end

    # Prior on η: ηᵀΩ⁻¹η  — use Cholesky solve for efficiency
    omega_chol = LowerTriangular(T.(params.omega.chol))
    eta_w      = omega_chol \ eta         # L⁻¹η
    nll += sum(abs2, eta_w)               # = ηᵀΩ⁻¹η

    # log|Ω| = 2 * Σᵢ log(Lᵢᵢ)
    nll += 2 * sum(log.(T.(diag(params.omega.chol))))

    return 0.5 * nll
end

# ---------------------------------------------------------------------------
# FOCE per-subject objective (linearized)
# ---------------------------------------------------------------------------

"""
    foce_subject_nll(eta_hat, H, subject, params, model)

FOCE (First-Order Conditional Estimation) negative log-likelihood contribution
for one subject, evaluated at the EBE η̂.

Uses first-order Taylor expansion of f around η̂:
  f̃(η) ≈ f(η̂) + H·(η - η̂)

Population prediction:  f̃(0) = f(η̂) - H·η̂
Linearized R̃ = diag(V(f̃(0))) + H·Ω·Hᵀ  (scalar diagonal V)

Note: R̃ is only diagonal when V depends on ipred and the error model is
proportional or additive. The full matrix form (H·Ω·Hᵀ + R) is used.

Returns a scalar T (Dual-compatible).
"""
function foce_subject_nll(eta_hat::Vector{Float64},
                            H::Matrix{Float64},
                            subject::Subject,
                            params::ModelParameters,
                            model::CompiledModel;
                            interaction::Bool = false)

    T = Float64
    ipred_hat = compute_predictions(model, subject,
                                     params.theta, eta_hat)

    # Linearized pop prediction (η = 0 in Taylor expansion)
    f0 = ipred_hat .- H * eta_hat     # f̃(0)

    # FOCE: R evaluated at population prediction f0
    # FOCE-I: R evaluated at individual prediction ipred_hat (eta-epsilon interaction)
    R_ref  = interaction ? ipred_hat : f0
    R_diag = compute_R_diag(model.error_model, R_ref, params.sigma.values)
    R = Diagonal(R_diag)

    # Total linearized covariance: R̃ = H·Ω·Hᵀ + R
    Ω = params.omega.matrix
    R_tilde = H * Ω * H' .+ Matrix(R)   # n×n

    y = subject.observations
    resid = y .- f0

    # Use Cholesky of R̃ for log-det and solve
    C_Rt = cholesky(Symmetric(R_tilde), check=false)
    if !issuccess(C_Rt)
        return Inf   # signal failure to optimizer
    end

    # (yᵢ - f̃₀)ᵀ R̃⁻¹ (yᵢ - f̃₀)
    quad_resid = dot(resid, C_Rt \ resid)

    # η̂ᵀ Ω⁻¹ η̂
    C_Ω = params.omega.chol
    eta_w = C_Ω \ eta_hat
    quad_eta = sum(abs2, eta_w)

    # log-determinants
    logdet_Rt = 2 * sum(log.(diag(C_Rt.L)))
    logdet_Ω  = 2 * sum(log.(diag(C_Ω)))

    return 0.5 * (quad_resid + logdet_Rt + quad_eta + logdet_Ω)
end

# ---------------------------------------------------------------------------
# Population FOCE objective (sum over all subjects)
# ---------------------------------------------------------------------------

"""
    foce_population_nll(params, population, model, eta_hats, H_mats)

Sum the FOCE NLL contributions over all subjects for given population
parameters. `eta_hats` and `H_mats` are held fixed (updated in inner loop).

This is the function whose gradient w.r.t. vectorized(θ, chol(Ω), log(σ))
is computed by the outer optimizer.
"""
function foce_population_nll(params::ModelParameters,
                               population::Population,
                               model::CompiledModel,
                               eta_hats::Vector{Vector{Float64}},
                               H_mats::Vector{Matrix{Float64}};
                               interaction::Bool = false)::Float64
    total = 0.0
    for (i, subject) in enumerate(population.subjects)
        total += foce_subject_nll(eta_hats[i], H_mats[i], subject, params, model;
                                   interaction)
    end
    return total
end

# ---------------------------------------------------------------------------
# AD-compatible variant for the outer gradient computation
# ---------------------------------------------------------------------------
# These take raw arrays (theta, omega_mat, sigma_vals) so ForwardDiff Dual
# numbers can flow through without hitting ModelParameters{Float64} walls.

"""
    foce_subject_nll_raw(eta_hat, H, subject, theta, omega_mat, sigma_vals, model)

AD-compatible version of `foce_subject_nll`. Accepts any `AbstractVector{T}` /
`AbstractMatrix{T}` so ForwardDiff can differentiate w.r.t. (theta, omega, sigma).
EBEs `eta_hat` and `H` are treated as fixed Float64 constants.
"""
function foce_subject_nll_raw(eta_hat::Vector{Float64},
                                H::Matrix{Float64},
                                subject::Subject,
                                theta::AbstractVector{T},
                                omega_mat::AbstractMatrix{T},
                                sigma_vals::AbstractVector{T},
                                model::CompiledModel;
                                interaction::Bool = false) where T<:Real

    ipred_hat = compute_predictions(model, subject, theta, T.(eta_hat))

    f0     = ipred_hat .- H * eta_hat
    R_ref  = interaction ? ipred_hat : f0
    R_diag = compute_R_diag(model.error_model, R_ref, sigma_vals)
    R      = Diagonal(R_diag)

    R_tilde = H * omega_mat * H' .+ Matrix(R)
    y     = subject.observations
    resid = T.(y) .- f0

    # Cholesky for log-det and solve; fall back to a large value if not PD
    R_tilde_sym = Symmetric(R_tilde)
    C_Rt = cholesky(R_tilde_sym, check=false)
    if !issuccess(C_Rt)
        return T(1e20)
    end

    quad_resid = dot(resid, C_Rt \ resid)
    logdet_Rt  = 2 * sum(log.(diag(C_Rt.L)))

    # η̂ᵀ Ω⁻¹ η̂  via Cholesky of omega_mat
    C_Ω = cholesky(Symmetric(omega_mat), check=false)
    if !issuccess(C_Ω)
        return T(1e20)
    end
    eta_w    = C_Ω.L \ T.(eta_hat)
    quad_eta = sum(abs2, eta_w)
    logdet_Ω = 2 * sum(log.(diag(C_Ω.L)))

    return T(0.5) * (quad_resid + logdet_Rt + quad_eta + logdet_Ω)
end

"""
    foce_population_nll_diff(theta, omega_mat, sigma_vals, population, model, eta_hats, H_mats)

AD-compatible population FOCE NLL. Used in the outer gradient computation with
EBEs fixed. Accepts Dual-typed arrays from ForwardDiff.
"""
function foce_population_nll_diff(theta::AbstractVector{T},
                                    omega_mat::AbstractMatrix{T},
                                    sigma_vals::AbstractVector{T},
                                    population::Population,
                                    model::CompiledModel,
                                    eta_hats::Vector{Vector{Float64}},
                                    H_mats::Vector{Matrix{Float64}};
                                    interaction::Bool = false) where T<:Real
    total = zero(T)
    for (i, subject) in enumerate(population.subjects)
        total += foce_subject_nll_raw(eta_hats[i], H_mats[i], subject,
                                       theta, omega_mat, sigma_vals, model;
                                       interaction)
    end
    return total
end
