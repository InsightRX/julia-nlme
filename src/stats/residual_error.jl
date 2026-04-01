"""
Residual error models.

Each model computes:
  - `residual_variance(model, ipred, sigma)` → scalar variance Var(y|η)
  - `weighted_residual(obs, ipred, model, sigma)` → (y - f) / √Var

All functions are AD-compatible (parametric in T<:Real).
"""

# ---------------------------------------------------------------------------
# Variance computation
# ---------------------------------------------------------------------------

"""
    residual_variance(error_model, ipred, sigma_values)

Return the residual variance Var(y_j | η_i) for one observation.

`sigma_values` is the `SigmaMatrix.values` vector.

Error models:
  :additive      →  σ₁²
  :proportional  →  (f·σ₁)²
  :combined      →  (f·σ₁)² + σ₂²
"""
function residual_variance(error_model::Symbol,
                            ipred::T,
                            sigma_values::AbstractVector) where T<:Real
    σ² = residual_variance_impl(Val(error_model), ipred, sigma_values)
    # Guard: variance must be positive (numerical issues near zero concentration)
    return max(σ², T(1e-12))
end

function residual_variance_impl(::Val{:additive}, ipred::T, σ) where T<:Real
    return T(σ[1])
end

function residual_variance_impl(::Val{:proportional}, ipred::T, σ) where T<:Real
    return (ipred * T(sqrt(σ[1])))^2
end

function residual_variance_impl(::Val{:combined}, ipred::T, σ) where T<:Real
    return (ipred * T(sqrt(σ[1])))^2 + T(σ[2])
end

# ---------------------------------------------------------------------------
# Diagonal R matrix for a subject (vector of variances)
# ---------------------------------------------------------------------------

"""
    compute_R_diag(error_model, ipreds, sigma_values)

Return the diagonal of the residual variance matrix R for a subject.
`ipreds` is the vector of individual predictions at observation times.
"""
function compute_R_diag(error_model::Symbol,
                         ipreds::AbstractVector{T},
                         sigma_values::AbstractVector) where T<:Real
    return [residual_variance(error_model, ip, sigma_values) for ip in ipreds]
end

# ---------------------------------------------------------------------------
# Individual weighted residuals
# ---------------------------------------------------------------------------

"""
    iwres(observations, ipreds, error_model, sigma_values)

Individual Weighted Residuals: (y - f̂) / √Var(y|η̂).
"""
function iwres(observations::AbstractVector{Float64},
               ipreds::AbstractVector{T},
               error_model::Symbol,
               sigma_values::AbstractVector) where T<:Real
    return [(observations[j] - ipreds[j]) / sqrt(residual_variance(error_model, ipreds[j], sigma_values))
            for j in eachindex(observations)]
end

# ---------------------------------------------------------------------------
# Conditional weighted residuals (CWRES) — computed from FOCE linearization
# ---------------------------------------------------------------------------

"""
    cwres(observations, pred, H, eta_hat, error_model, sigma_values, omega)

Conditional Weighted Residuals using the FOCE linearized model.

  f̃(η=0) = f(η̂) - H·η̂   (population prediction from linearized model)
  R̃ = H·Ω·Hᵀ + diag(R)  (total variance from between- and within-subject)
  CWRES_j = (y_j - f̃_j) / √R̃_jj
"""
function cwres(observations::AbstractVector{Float64},
               ipreds::AbstractVector{Float64},
               H::AbstractMatrix{Float64},
               eta_hat::AbstractVector{Float64},
               error_model::Symbol,
               sigma_values::AbstractVector{Float64},
               omega::AbstractMatrix{Float64})

    n = length(observations)
    # Linearized population predictions
    f_tilde_zero = ipreds .- H * eta_hat

    # Total residual covariance matrix
    R_diag = compute_R_diag(error_model, ipreds, sigma_values)
    R = Diagonal(R_diag)
    R_tilde = H * omega * H' .+ R  # n×n matrix

    residuals = observations .- f_tilde_zero
    R_tilde_diag = diag(R_tilde)
    return [residuals[j] / sqrt(max(R_tilde_diag[j], 1e-12)) for j in 1:n]
end
