"""
Likelihood functions for NLME estimation.

The individual log-likelihood (Laplacian approximation) is used in the inner
η-optimization. The FOCE approximation of the population objective function
is used in the outer optimization over θ, Ω, σ.

FOCE (no interaction, interaction=false) — paper eq. 15:
  NLL = 0.5 × [(y-f0)ᵀ R̃⁻¹ (y-f0) + log|R̃|]
  where f0 = f(η̂) - H·η̂,  R̃ = H Ω Hᵀ + R(f0)

FOCEI (interaction=true) — paper eq. 20, simplified via matrix determinant lemma:
  NLL = 0.5 × [(y-IPRED)ᵀ V⁻¹ (y-IPRED) + η̂ᵀΩ⁻¹η̂ + log|R̃|]
  where IPRED = f(η̂),  V = R(IPRED),  R̃ = H Ω Hᵀ + V

The simplification uses the identity:
  log|Ω| + log|Φᵢ| + log|Ω⁻¹ + HᵀV⁻¹H| = log|HΩHᵀ + V| = log|R̃|
(matrix determinant lemma), so the three Ω-dependent terms in eq. 20 reduce to
log|R̃|, and Φᵢ = log|V| + (y-IPRED)ᵀV⁻¹(y-IPRED) contributes its log|V|
which cancels with the -log|V| from the lemma, leaving only
(y-IPRED)ᵀV⁻¹(y-IPRED). The remaining explicit Ω term is η̂ᵀΩ⁻¹η̂, which
(together with log|R̃|) prevents Ω from collapsing to zero.

OFV constant for both approximations:
  OFV = 2 × NLL + n_obs × log(2π)
The n_eta × log(2π) from the η prior and from the Laplace correction cancel
exactly, so only the n_obs observation dimension contributes.
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
# FOCE / FOCEI per-subject objective
# ---------------------------------------------------------------------------

"""
    foce_subject_nll(eta_hat, H, subject, params, model; interaction)

FOCE or FOCEI negative log-likelihood contribution for one subject.

**FOCE (interaction=false)** — linearized marginal likelihood (paper eq. 15):
  NLL = 0.5 × [(y-f0)ᵀ R̃⁻¹ (y-f0) + log|R̃|]

**FOCEI (interaction=true)** — NONMEM Laplace / FOCEI (paper eq. 20),
simplified via the matrix determinant lemma:
  NLL = 0.5 × [(y-IPRED)ᵀ V⁻¹ (y-IPRED) + η̂ᵀΩ⁻¹η̂ + log|R̃|]

See module docstring for the simplification derivation.

Both formulations use R̃ = H Ω Hᵀ + V where V is the diagonal residual
variance evaluated at the appropriate reference point.
"""
function foce_subject_nll(eta_hat::Vector{Float64},
                            H::Matrix{Float64},
                            subject::Subject,
                            params::ModelParameters,
                            model::CompiledModel;
                            interaction::Bool = false)

    ipred_hat = compute_predictions(model, subject, params.theta, eta_hat)
    Ω = params.omega.matrix

    if interaction
        # FOCEI: R evaluated at individual prediction
        V_diag  = compute_R_diag(model.error_model, ipred_hat, params.sigma.values)
        R_tilde = H * Ω * H' .+ Matrix(Diagonal(V_diag))

        C_Rt = cholesky(Symmetric(R_tilde), check=false)
        issuccess(C_Rt) || return Inf
        logdet_Rt = 2 * sum(log.(diag(C_Rt.L)))

        # Non-linearized residual from IPRED
        y        = subject.observations
        resid    = y .- ipred_hat
        quad_obs = dot(resid, resid ./ V_diag)

        # η̂ᵀΩ⁻¹η̂ — params.omega.chol is already LowerTriangular
        eta_w    = params.omega.chol \ eta_hat
        quad_eta = sum(abs2, eta_w)

        return 0.5 * (quad_obs + quad_eta + logdet_Rt)
    else
        # FOCE: R evaluated at linearized population prediction f0
        f0     = ipred_hat .- H * eta_hat
        V_diag = compute_R_diag(model.error_model, f0, params.sigma.values)
        R_tilde = H * Ω * H' .+ Matrix(Diagonal(V_diag))

        C_Rt = cholesky(Symmetric(R_tilde), check=false)
        issuccess(C_Rt) || return Inf

        y     = subject.observations
        resid = y .- f0
        quad_resid = dot(resid, C_Rt \ resid)
        logdet_Rt  = 2 * sum(log.(diag(C_Rt.L)))

        return 0.5 * (quad_resid + logdet_Rt)
    end
end

# ---------------------------------------------------------------------------
# Population FOCE objective (sum over all subjects)
# ---------------------------------------------------------------------------

"""
    foce_population_nll(params, population, model, eta_hats, H_mats; interaction)

Sum the FOCE/FOCEI NLL contributions over all subjects for given population
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
# AD-compatible variants for the outer gradient computation
# ---------------------------------------------------------------------------
# These take raw arrays (theta, omega_mat, sigma_vals) so ForwardDiff Dual
# numbers can flow through without hitting ModelParameters{Float64} walls.

"""
    foce_subject_nll_raw(eta_hat, H, subject, theta, omega_mat, sigma_vals, model; interaction)

AD-compatible FOCE/FOCEI NLL. Accepts any `AbstractVector{T}` / `AbstractMatrix{T}`
so ForwardDiff can differentiate w.r.t. (theta, omega, sigma).
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

    if interaction
        # FOCEI (simplified): 0.5 × [(y-IPRED)ᵀV⁻¹(y-IPRED) + η̂ᵀΩ⁻¹η̂ + log|R̃|]
        V_diag  = compute_R_diag(model.error_model, ipred_hat, sigma_vals)
        R_tilde = H * omega_mat * H' .+ Matrix(Diagonal(V_diag))

        R_tilde_sym = Symmetric(R_tilde)
        C_Rt = cholesky(R_tilde_sym, check=false)
        if !issuccess(C_Rt)
            return T(1e20)
        end
        logdet_Rt = 2 * sum(log.(diag(C_Rt.L)))

        y        = subject.observations
        resid    = T.(y) .- ipred_hat
        quad_obs = dot(resid, resid ./ V_diag)

        # η̂ᵀΩ⁻¹η̂ — use LU solve (omega_mat \ eta_hat) rather than a second
        # Cholesky factorisation. cholesky() on Dual-typed matrices can silently
        # fail issuccess() in some Julia versions, producing a zero gradient.
        # LU-based \ is reliably differentiable through ForwardDiff.
        eta_w    = omega_mat \ T.(eta_hat)
        quad_eta = dot(T.(eta_hat), eta_w)

        return T(0.5) * (quad_obs + quad_eta + logdet_Rt)
    else
        # FOCE (linearized)
        f0     = ipred_hat .- H * eta_hat
        V_diag = compute_R_diag(model.error_model, f0, sigma_vals)
        R_tilde = H * omega_mat * H' .+ Matrix(Diagonal(V_diag))

        R_tilde_sym = Symmetric(R_tilde)
        C_Rt = cholesky(R_tilde_sym, check=false)
        if !issuccess(C_Rt)
            return T(1e20)
        end

        y     = subject.observations
        resid = T.(y) .- f0
        quad_resid = dot(resid, C_Rt \ resid)
        logdet_Rt  = 2 * sum(log.(diag(C_Rt.L)))

        return T(0.5) * (quad_resid + logdet_Rt)
    end
end

"""
    foce_population_nll_diff(theta, omega_mat, sigma_vals, population, model, eta_hats, H_mats; interaction)

AD-compatible population FOCE/FOCEI NLL. Used in the outer gradient computation with
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
