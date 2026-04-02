"""
Importance Sampling (IS) marginal likelihood for NLME.

Computes an unbiased Monte Carlo estimate of the marginal log-likelihood:

  log p(yᵢ; θ) = log ∫ p(yᵢ|ηᵢ; θ,σ) p(ηᵢ; Ω) dηᵢ
              ≈ log [ (1/M) Σₘ p(yᵢ|ηᵢ^(m)) p(ηᵢ^(m)) / q(ηᵢ^(m)) ]

Proposal: Laplace approximation of the posterior p(ηᵢ|yᵢ),
  q(ηᵢ) = N(η̂ᵢ, Σᵢ)
where η̂ᵢ is the EBE and Σᵢ = [∂² individual_nll / ∂η²]⁻¹ is the
posterior covariance from the Hessian of the Laplace objective.

Log-weight formula (NONMEM normalisation — omits the n_obs × log(2π)
constant so that OFV_IS is directly comparable with OFV_FOCE):

  log wᵢ^(m) = −individual_nll(ηᵢ^(m)) + ½‖z^(m)‖² + log|L_Σ|ᵢ

where ηᵢ^(m) = η̂ᵢ + L_Σᵢ z^(m),  z^(m) ∼ N(0,I).

The Laplace / FOCE approximation that FOCE uses should give OFV_IS ≤ OFV_FOCE
(IS is a lower bound on FOCE; a large gap suggests the Laplace approximation
is inaccurate for that model/dataset).
"""

using Random, LinearAlgebra, ForwardDiff, Printf
using Statistics: mean

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

"""
    ISResult

Marginal log-likelihood estimate from importance sampling.

Fields:
  `ofv`               IS estimate of −2 log L (NONMEM convention)
  `aic`, `bic`        Information criteria using IS OFV
  `foce_ofv`          Original FOCE OFV for comparison
  `delta_ofv`         foce_ofv − ofv (positive = FOCE was conservative)
  `per_subject_loglik` Per-subject log p(yᵢ; θ) estimates
  `ess`               Per-subject effective sample size (max = n_samples)
  `ess_pct`           ESS as percentage of n_samples
  `n_samples`         M used
  `theta`, `omega`, `sigma`   Parameter estimates (from the supplied FitResult)
"""
struct ISResult
    ofv::Float64
    aic::Float64
    bic::Float64
    foce_ofv::Float64
    delta_ofv::Float64

    per_subject_loglik::Vector{Float64}
    ess::Vector{Float64}
    ess_pct::Vector{Float64}
    n_samples::Int

    theta::Vector{Float64}
    theta_names::Vector{Symbol}
    omega::Matrix{Float64}
    sigma::Vector{Float64}

    n_obs::Int
    n_subjects::Int
    n_parameters::Int
end

function Base.show(io::IO, r::ISResult)
    println(io, "ISResult")
    println(io, "  OFV (IS):   $(round(r.ofv,   digits=3))  " *
                "  AIC: $(round(r.aic, digits=3))  " *
                "  BIC: $(round(r.bic, digits=3))")
    println(io, "  OFV (FOCE): $(round(r.foce_ofv, digits=3))  " *
                "  ΔOFV: $(round(r.delta_ofv, digits=3))" *
                (r.delta_ofv > 0 ? "  (FOCE was conservative)" : ""))
    println(io, "  Samples: $(r.n_samples)   " *
                "Subjects: $(r.n_subjects)   Observations: $(r.n_obs)")
    println(io, "  Mean ESS: $(round(mean(r.ess_pct), digits=1))%  " *
                "Min ESS: $(round(minimum(r.ess_pct), digits=1))%")
end

# ---------------------------------------------------------------------------
# Build the Laplace proposal for one subject
# ---------------------------------------------------------------------------

"""
    _proposal_chol(eta_hat, subject, model, params)

Compute L_Σ, the lower-Cholesky factor of the proposal covariance
Σ = [∂² individual_nll / ∂η²]⁻¹  evaluated at η̂.

Falls back to chol(Ω) if the Hessian is not positive definite
(e.g. EBE not fully converged).
"""
function _proposal_chol(eta_hat::Vector{Float64},
                          subject::Subject,
                          model::CompiledModel,
                          params::ModelParameters)
    H = try
        ForwardDiff.hessian(
            η -> individual_nll(η, subject, params, model),
            eta_hat)
    catch
        nothing
    end

    if H !== nothing
        C = cholesky(Symmetric(H), check = false)
        issuccess(C) && return LowerTriangular(collect(C.L))
    end

    # Fallback: use Ω as proposal covariance (still valid, just less efficient)
    return params.omega.chol
end

# ---------------------------------------------------------------------------
# Per-subject importance sampling
# ---------------------------------------------------------------------------

"""
    _is_subject(subject, model, params, eta_hat, proposal_chol, n_samples, rng)

Estimate log p(yᵢ; θ) and the effective sample size for one subject.

Returns `(log_p_yi, ess)`.
"""
function _is_subject(subject::Subject,
                      model::CompiledModel,
                      params::ModelParameters,
                      eta_hat::Vector{Float64},
                      proposal_chol::LowerTriangular{Float64},
                      n_samples::Int,
                      rng::Random.AbstractRNG)
    d           = length(eta_hat)
    log_detL    = sum(log.(diag(proposal_chol)))   # log|L_Σ| = ½ log|Σ|
    log_weights = Vector{Float64}(undef, n_samples)

    for m in 1:n_samples
        z     = randn(rng, d)
        eta_m = eta_hat .+ proposal_chol * z

        # log w^(m) = −individual_nll(η^(m)) + ½‖z‖² + log|L_Σ|
        # (uses NONMEM convention: omits n_obs × log(2π); see module docstring)
        nll_m          = individual_nll(eta_m, subject, params, model)
        log_weights[m] = -nll_m + 0.5 * dot(z, z) + log_detL
    end

    # log p̂(yᵢ) = logsumexp(log w) − log(M)
    log_p_yi = logsumexp(log_weights) - log(n_samples)

    # ESS = exp(2 logsumexp(log w) − logsumexp(2 log w))
    ess = exp(2 * logsumexp(log_weights) - logsumexp(2 .* log_weights))

    return log_p_yi, ess
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    importance_sampling(result, population; n_samples=500, rng=Random.default_rng())

Estimate the marginal log-likelihood via importance sampling using the
parameter estimates and EBEs from a `FitResult` (FOCE or SAEM).

The Laplace posterior at each EBE is used as the proposal distribution.
Returns an `ISResult` with the IS OFV, per-subject log-likelihoods, and
effective sample sizes (ESS). A low ESS (<10%) flags subjects where the
Laplace approximation is poor.

# Example
```julia
result_foce = fit(model, pop; interaction=true)
result_saem = fit_saem(model, pop)

is_foce = importance_sampling(result_foce, pop; n_samples=1000)
is_saem = importance_sampling(result_saem, pop; n_samples=1000)
println(is_foce)
println(is_saem)
```
"""
function importance_sampling(result::FitResult,
                               population::Population;
                               n_samples::Int              = 500,
                               rng::Random.AbstractRNG     = Random.default_rng(),
                               verbose::Bool               = true)

    model = result.model

    # Reconstruct ModelParameters from FitResult
    template = model.default_params
    omega    = OmegaMatrix(result.omega, template.omega.eta_names;
                            diagonal = template.omega.diagonal)
    params   = ModelParameters(result.theta, template.theta_names,
                                template.theta_lower, template.theta_upper,
                                omega,
                                SigmaMatrix(result.sigma, template.sigma.names),
                                template.packed_fixed)

    N     = length(population)
    n_obs = result.n_obs

    verbose && @info @sprintf(
        "Importance sampling: %d subjects, %d samples each", N, n_samples)

    per_subject_loglik = zeros(N)
    ess                = zeros(N)

    # Per-thread RNGs for thread safety
    # Use maxthreadid() (not nthreads()) — Julia 1.9+ has multiple thread pools
    # and threadid() can exceed nthreads() for the default pool.
    thread_rngs = [Random.MersenneTwister(rand(rng, UInt32))
                   for _ in 1:Threads.maxthreadid()]

    Threads.@threads for i in 1:N
        tid     = Threads.threadid()
        subject = population.subjects[i]
        eta_hat = result.subjects[i].eta

        prop_chol = _proposal_chol(eta_hat, subject, model, params)

        log_p_yi, ess_i = _is_subject(subject, model, params, eta_hat,
                                        prop_chol, n_samples, thread_rngs[tid])
        per_subject_loglik[i] = log_p_yi
        ess[i]                = ess_i
    end

    ofv        = -2 * sum(per_subject_loglik)
    n_params   = result.n_parameters
    aic        = ofv + 2 * n_params
    bic        = ofv + n_params * log(n_obs)
    ess_pct    = 100 .* ess ./ n_samples
    delta_ofv  = result.ofv - ofv

    verbose && @info @sprintf(
        "IS OFV = %.3f  ΔOFV vs FOCE = %.3f  mean ESS = %.1f%%",
        ofv, delta_ofv, mean(ess_pct))

    return ISResult(ofv, aic, bic, result.ofv, delta_ofv,
                    per_subject_loglik, ess, ess_pct, n_samples,
                    result.theta, result.theta_names,
                    result.omega, result.sigma,
                    n_obs, N, n_params)
end
