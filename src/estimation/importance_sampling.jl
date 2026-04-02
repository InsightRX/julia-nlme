"""
Importance Sampling (IS) parameter uncertainty estimation for NLME.

Samples M packed parameter vectors from N(x̂, Ĉ) using the FOCE covariance
matrix Ĉ, evaluates the FOCE population NLL at each sample, and computes
IS weights:

  log w^(m) = −NLL_FOCE(x^(m)) + ½‖z^(m)‖²

where x^(m) = x̂_free + L_C z^(m), z^(m) ∼ N(0, I), L_C = chol(Ĉ_free).

The normalized weights {w̃^(m)} define a weighted distribution over population
parameters from which IS-based SE and non-parametric 95% CI are derived for
all parameters (THETA, OMEGA diagonal, SIGMA).

IS OFV:
  OFV_IS = −2 (logsumexp(log w) − log M)

This marginalises the FOCE likelihood over parameter uncertainty, giving a
model fit statistic that is directly comparable to OFV_FOCE. A positive ΔOFV
(FOCE − IS > 0) means the Laplace / quadratic approximation was conservative.
"""

using Random, LinearAlgebra, Printf
using Statistics: mean, std

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

"""
    ISResult

Importance sampling result with IS-based OFV and parameter uncertainty.

Fields:
  `ofv`, `aic`, `bic`      IS estimate of −2 log L (integrating over parameter uncertainty)
  `foce_ofv`, `delta_ofv`  Original FOCE OFV and difference (FOCE − IS)
  `ess`, `ess_pct`         Effective sample size over parameter samples
  `n_samples`              M used
  `theta`, `omega`, `sigma`   Point estimates (from the source FitResult)
  `se_theta/omega/sigma`   IS-based SE (weighted std of parameter samples)
  `ci_theta/omega/sigma`   Non-parametric 95% CI (2.5th–97.5th weighted percentile)
                           omega fields cover diagonal variance elements only
"""
struct ISResult
    ofv::Float64
    aic::Float64
    bic::Float64
    foce_ofv::Float64
    delta_ofv::Float64

    ess::Float64
    ess_pct::Float64
    n_samples::Int

    theta::Vector{Float64}
    theta_names::Vector{Symbol}
    omega::Matrix{Float64}
    sigma::Vector{Float64}

    # IS-based uncertainty (weighted statistics over parameter samples)
    se_theta::Vector{Float64}
    se_omega::Vector{Float64}          # diagonal variance elements only
    se_sigma::Vector{Float64}

    ci_theta::Vector{NTuple{2, Float64}}   # (2.5th, 97.5th percentile)
    ci_omega::Vector{NTuple{2, Float64}}   # diagonal variance elements only
    ci_sigma::Vector{NTuple{2, Float64}}

    n_obs::Int
    n_subjects::Int
    n_parameters::Int
end

function Base.show(io::IO, r::ISResult)
    println(io, "\n" * "="^72)
    println(io, "  Importance Sampling Results")
    println(io, "="^72)
    @printf io "  OFV (IS):   %10.3f    AIC: %10.3f    BIC: %10.3f\n" r.ofv r.aic r.bic
    delta_str = r.delta_ofv > 0 ? "  (FOCE conservative)" : ""
    @printf io "  OFV (FOCE): %10.3f    ΔOFV: %+.3f%s\n" r.foce_ofv r.delta_ofv delta_str
    @printf io "  Subjects: %d    Observations: %d    Samples: %d    ESS: %.1f%%\n" r.n_subjects r.n_obs r.n_samples r.ess_pct

    sep = "-"^82

    function _print_rows(names, estimates, ses, cis)
        println(io, "  " * sep)
        @printf io "  %-18s  %12s  %10s  %8s  %24s\n" "Name" "Estimate" "SE" "%RSE" "95% CI"
        println(io, "  " * sep)
        for (name, est, se, (ci_lo, ci_hi)) in zip(names, estimates, ses, cis)
            rse    = est != 0 ? abs(se / est) * 100 : NaN
            ci_str = @sprintf "[%9.4g, %9.4g]" ci_lo ci_hi
            @printf io "  %-18s  %12.4g  %10.4g  %7.1f%%  %s\n" name est se rse ci_str
        end
    end

    println(io, "\n  THETA:")
    _print_rows(string.(r.theta_names), r.theta, r.se_theta, r.ci_theta)

    n_eta = size(r.omega, 1)
    println(io, "\n  OMEGA (diagonal variance):")
    _print_rows(["OMEGA($i,$i)" for i in 1:n_eta],
                [r.omega[i,i]   for i in 1:n_eta],
                r.se_omega, r.ci_omega)

    println(io, "\n  SIGMA:")
    _print_rows(["SIGMA($i)" for i in 1:length(r.sigma)],
                r.sigma, r.se_sigma, r.ci_sigma)
    println(io, "  " * sep)
end

# ---------------------------------------------------------------------------
# Weighted statistics helpers
# ---------------------------------------------------------------------------

"""Weighted standard deviation: sqrt(Σ w̃ (x − x̄)²)."""
function _weighted_se(samples::AbstractVector{Float64},
                       weights::AbstractVector{Float64})
    μ = sum(weights .* samples)
    sqrt(max(0.0, sum(weights .* (samples .- μ).^2)))
end

"""Non-parametric CI: 2.5th and 97.5th percentile of the weighted distribution."""
function _weighted_ci(samples::AbstractVector{Float64},
                       weights::AbstractVector{Float64})
    ord    = sortperm(samples)
    cumw   = cumsum(weights[ord])
    s_sort = samples[ord]
    lo = s_sort[something(findfirst(>=(0.025), cumw), lastindex(cumw))]
    hi = s_sort[something(findfirst(>=(0.975), cumw), lastindex(cumw))]
    return (lo, hi)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    importance_sampling(result, population; n_samples=500, rng=Random.default_rng(), verbose=true)

Estimate parameter uncertainty via importance sampling using the parameter
estimates and covariance matrix from a `FitResult` (FOCE or SAEM).

M packed parameter vectors are sampled from N(x̂, Ĉ) where Ĉ is the FOCE
covariance matrix. The FOCE population likelihood is evaluated at each sample
to compute IS weights. The resulting weighted distribution gives IS-based SE
and non-parametric 95% CI for all population parameters.

Requires `run_covariance_step=true` when calling `fit()` or `fit_saem()`.

# Example
```julia
result = fit(model, pop; interaction=true)
is = importance_sampling(result, pop; n_samples=1000)
println(is)
```
"""
function importance_sampling(result::FitResult,
                               population::Population;
                               n_samples::Int          = 500,
                               rng::Random.AbstractRNG = Random.default_rng(),
                               verbose::Bool           = true)

    if isempty(result.covariance_matrix)
        error("Importance sampling requires a parameter covariance matrix. " *
              "Re-run fit() or fit_saem() with run_covariance_step=true.")
    end

    model    = result.model
    template = model.default_params

    # Reconstruct ModelParameters and packed parameter vector from FitResult
    omega  = OmegaMatrix(result.omega, template.omega.eta_names;
                          diagonal = template.omega.diagonal)
    params = ModelParameters(result.theta, template.theta_names,
                              template.theta_lower, template.theta_upper,
                              omega,
                              SigmaMatrix(result.sigma, template.sigma.names),
                              template.packed_fixed)
    x_hat = pack_params(params)

    # Free parameter indices (same exclusion as in optimize_population)
    free_idx = isempty(template.packed_fixed) ?
        collect(1:length(x_hat)) :
        findall(.!template.packed_fixed)

    x_hat_free = x_hat[free_idx]
    C_free     = result.covariance_matrix[free_idx, free_idx]
    L_C        = cholesky(Symmetric(C_free)).L   # for drawing N(0,I) → N(x̂, Ĉ)

    function expand(x_free)
        v = copy(x_hat)
        v[free_idx] .= x_free
        return v
    end

    N     = length(population)
    n_obs = result.n_obs

    # Build FOCE objective for likelihood evaluation at each parameter sample.
    # verbose=false: suppresses per-evaluation progress messages.
    f_only, _, _, _ = make_outer_objective(
        population, model, template;
        interaction = result.interaction,
        verbose     = false)

    verbose && @info @sprintf(
        "Importance sampling: %d parameter samples, %d subjects", n_samples, N)

    # -------------------------------------------------------------------------
    # Draw parameter samples and evaluate FOCE likelihood at each
    # -------------------------------------------------------------------------
    n_free       = length(x_hat_free)
    n_theta      = length(result.theta)
    n_eta        = size(result.omega, 1)
    n_sigma      = length(result.sigma)

    log_weights  = Vector{Float64}(undef, n_samples)
    theta_mat    = Matrix{Float64}(undef, n_theta, n_samples)
    omega_d_mat  = Matrix{Float64}(undef, n_eta,   n_samples)  # diagonal Ω only
    sigma_mat    = Matrix{Float64}(undef, n_sigma,  n_samples)

    for m in 1:n_samples
        z        = randn(rng, n_free)
        x_free_m = x_hat_free .+ L_C * z
        x_m      = expand(x_free_m)

        # NLL_FOCE runs the inner EBE optimization for all subjects at x_m.
        nll_m = try
            f_only(x_m)
        catch
            Inf
        end

        log_weights[m] = -nll_m + 0.5 * dot(z, z)

        # Unpack natural-space parameters for this sample
        pm = unpack_params(x_m, template)
        theta_mat[:, m]   = pm.theta
        omega_d_mat[:, m] = [pm.omega.matrix[i, i] for i in 1:n_eta]
        sigma_mat[:, m]   = pm.sigma.values
    end

    # -------------------------------------------------------------------------
    # IS OFV and ESS
    # -------------------------------------------------------------------------
    log_lse = logsumexp(log_weights)
    ofv     = -2.0 * (log_lse - log(n_samples))
    n_params = result.n_parameters
    aic      = ofv + 2 * n_params
    bic      = ofv + n_params * log(n_obs)
    delta_ofv = result.ofv - ofv

    ess     = exp(2 * log_lse - logsumexp(2 .* log_weights))
    ess_pct = 100.0 * ess / n_samples

    # Normalise weights for computing weighted statistics
    w = exp.(log_weights .- log_lse)

    # -------------------------------------------------------------------------
    # Weighted SE and non-parametric CI for each parameter
    # -------------------------------------------------------------------------
    se_theta = [_weighted_se(theta_mat[i, :],   w) for i in 1:n_theta]
    se_omega = [_weighted_se(omega_d_mat[i, :], w) for i in 1:n_eta]
    se_sigma = [_weighted_se(sigma_mat[i, :],   w) for i in 1:n_sigma]

    ci_theta = [_weighted_ci(theta_mat[i, :],   w) for i in 1:n_theta]
    ci_omega = [_weighted_ci(omega_d_mat[i, :], w) for i in 1:n_eta]
    ci_sigma = [_weighted_ci(sigma_mat[i, :],   w) for i in 1:n_sigma]

    verbose && @info @sprintf(
        "IS OFV = %.3f  ΔOFV vs FOCE = %.3f  ESS = %.1f%%",
        ofv, delta_ofv, ess_pct)

    return ISResult(ofv, aic, bic, result.ofv, delta_ofv,
                    ess, ess_pct, n_samples,
                    result.theta, result.theta_names,
                    result.omega, result.sigma,
                    se_theta, se_omega, se_sigma,
                    ci_theta, ci_omega, ci_sigma,
                    n_obs, N, n_params)
end
