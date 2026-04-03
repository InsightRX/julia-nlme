"""
Core data structures for JuliaNLME.
"""

# ---------------------------------------------------------------------------
# Dose / Event types
# ---------------------------------------------------------------------------

"""
A single dosing event for one subject.

`rate == 0.0` means bolus; `rate > 0` means constant-rate infusion.
`duration` is derived from `amt/rate` when rate > 0.
`ss == true` triggers steady-state pre-dosing.
"""
struct DoseEvent
    time::Float64
    amt::Float64
    cmt::Int
    rate::Float64      # 0.0 = bolus
    duration::Float64  # 0.0 for bolus
    ss::Bool
    ii::Float64        # inter-dose interval (for SS)
end

DoseEvent(time, amt; cmt=1, rate=0.0, ss=false, ii=0.0) =
    DoseEvent(time, amt, cmt, rate,
              rate > 0 ? amt / rate : 0.0,
              ss, ii)

# ---------------------------------------------------------------------------
# Subject
# ---------------------------------------------------------------------------

"""
All data for one individual in the population.
"""
struct Subject
    id::Int
    doses::Vector{DoseEvent}
    obs_times::Vector{Float64}    # times of non-missing observations
    observations::Vector{Float64} # DV values at obs_times
    obs_cmts::Vector{Int}         # compartment for each observation
    covariates::Dict{Symbol, Float64}          # time-constant covariates
    tvcov::Dict{Symbol, Vector{Float64}}       # time-varying: name → value at each obs_time (LOCF)
end

# Backward-compatible constructor (no tvcov) — used in tests and direct construction
Subject(id, doses, obs_times, observations, obs_cmts, covariates) =
    Subject(id, doses, obs_times, observations, obs_cmts, covariates,
            Dict{Symbol, Vector{Float64}}())

Base.show(io::IO, s::Subject) =
    print(io, "Subject(id=$(s.id), ndose=$(length(s.doses)), nobs=$(length(s.observations)))")

# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

"""
The full dataset as a collection of subjects.
"""
struct Population
    subjects::Vector{Subject}
    covariate_names::Vector{Symbol}
    dv_column::Symbol
end

Base.length(p::Population) = length(p.subjects)
Base.iterate(p::Population, state=1) = state > length(p.subjects) ? nothing : (p.subjects[state], state+1)
Base.getindex(p::Population, i) = p.subjects[i]

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
Variance-covariance matrix for between-subject variability (OMEGA).

`chol` is the lower-triangular Cholesky factor; `matrix = chol * chol'`.
When `diagonal = true`, off-diagonal elements are fixed at zero and only the
`n_eta` diagonal variances are estimated during optimisation.
"""
struct OmegaMatrix
    matrix::Matrix{Float64}
    chol::LowerTriangular{Float64, Matrix{Float64}}
    eta_names::Vector{Symbol}
    diagonal::Bool
end

# Convenience constructors (diagonal defaults to false — full covariance)
OmegaMatrix(matrix::Matrix{Float64},
            chol::LowerTriangular{Float64, Matrix{Float64}},
            eta_names::Vector{Symbol}) =
    OmegaMatrix(matrix, chol, eta_names, false)

function OmegaMatrix(matrix::AbstractMatrix, eta_names::Vector{Symbol};
                     diagonal::Bool = false)
    m = collect(Symmetric(matrix))
    # During line search, extreme parameter values can produce Inf/NaN matrices.
    # Fall back to a small identity so the optimizer gets a finite (large) OFV
    # and backtracks rather than crashing.
    if !all(isfinite, m)
        n    = size(m, 1)
        m_fb = Matrix{Float64}(I, n, n) .* 1e-4
        L_fb = LowerTriangular(Matrix{Float64}(I, n, n) .* sqrt(1e-4))
        return OmegaMatrix(m_fb, L_fb, eta_names, diagonal)
    end
    C = cholesky(Symmetric(m), check=false)
    if issuccess(C)
        return OmegaMatrix(m, LowerTriangular(collect(C.L)), eta_names, diagonal)
    end
    # Near-singular: regularize so optimizer can continue
    λ_min = minimum(eigvals(Symmetric(m)))
    reg = max(-λ_min + 1e-8, 1e-8)
    m_reg = m + reg * I
    C_reg = cholesky(Symmetric(m_reg))
    OmegaMatrix(m_reg, LowerTriangular(collect(C_reg.L)), eta_names, diagonal)
end

OmegaMatrix(diag_vals::Vector{Float64}, eta_names::Vector{Symbol};
            diagonal::Bool = false) =
    OmegaMatrix(Diagonal(diag_vals) |> Matrix, eta_names; diagonal)

n_etas(ω::OmegaMatrix) = length(ω.eta_names)

"""
Diagonal residual variance matrix (SIGMA).
"""
struct SigmaMatrix
    values::Vector{Float64}  # variance parameters (σ₁², σ₂², ...)
    names::Vector{Symbol}
end

"""
Complete set of population parameters at a given iteration.
"""
struct ModelParameters
    theta::Vector{Float64}
    theta_names::Vector{Symbol}
    theta_lower::Vector{Float64}   # lower bounds for each theta (enforced via Fminbox)
    theta_upper::Vector{Float64}   # upper bounds for each theta
    omega::OmegaMatrix
    sigma::SigmaMatrix
    # Which elements of the packed parameter vector are fixed (lower == upper == x0).
    # Empty vector means nothing is fixed (backward-compatible default).
    packed_fixed::Vector{Bool}
end

# Backward-compatible constructors
ModelParameters(theta, theta_names, lower, upper, omega, sigma) =
    ModelParameters(theta, theta_names, lower, upper, omega, sigma, Bool[])

ModelParameters(theta, theta_names, omega, sigma) =
    ModelParameters(theta, theta_names,
                    fill(1e-9, length(theta)),
                    fill(1e9,  length(theta)),
                    omega, sigma, Bool[])

# ---------------------------------------------------------------------------
# Compiled model (produced by parser)
# ---------------------------------------------------------------------------

"""
ODE specification for models defined with the `[odes]` block.

  - `ode_fn`: in-place ODE function `(du, u, p, t)` where `p` is the
    NamedTuple returned by `pk_param_fn`. Must be generic over `T<:Real`
    so ForwardDiff dual numbers flow through the ODE solve.
  - `state_names`: ordered list of state variable names (matches u indices)
  - `obs_cmt_idx`: index into `state_names` of the observable compartment
"""
struct ODESpec
    ode_fn::Function
    state_names::Vector{Symbol}
    obs_cmt_idx::Int
end

"""
A model ready for estimation. The parser (or user) produces this by
providing Julia functions for each model component.

  - `pk_param_fn(theta, eta, covariates) → NamedTuple` of individual PK params
  - `pk_model`: one of the symbols registered in PKEquations, or `:ode`
  - `error_model`: :additive | :proportional | :combined
  - `ode_spec`: non-nothing for ODE models; `nothing` for analytical models
"""
struct CompiledModel
    name::String
    pk_model::Symbol
    error_model::Symbol

    # Generated functions
    pk_param_fn::Function   # (theta, eta, cov) → NamedTuple

    # Metadata
    n_theta::Int
    n_eta::Int
    n_epsilon::Int
    theta_names::Vector{Symbol}
    eta_names::Vector{Symbol}

    # Default initial parameters built from the [parameters] block.
    # Used by fit(model, population) when no init_params are supplied.
    default_params::ModelParameters

    # ODE specification (nothing for analytical models)
    ode_spec::Union{Nothing, ODESpec}
end

# Backward-compatible constructor for analytical models (no ode_spec)
CompiledModel(name, pk_model, error_model, pk_param_fn, n_theta, n_eta, n_epsilon,
              theta_names, eta_names, default_params) =
    CompiledModel(name, pk_model, error_model, pk_param_fn, n_theta, n_eta, n_epsilon,
                  theta_names, eta_names, default_params, nothing)

# ---------------------------------------------------------------------------
# Estimation results
# ---------------------------------------------------------------------------

"""
Per-subject post-hoc results.
"""
struct SubjectResult
    id::Int
    eta::Vector{Float64}       # EBEs
    ipred::Vector{Float64}     # individual predictions
    pred::Vector{Float64}      # population predictions (eta = 0)
    iwres::Vector{Float64}     # individual weighted residuals
    cwres::Vector{Float64}     # conditional weighted residuals
    ofv_contribution::Float64
end

"""
Full result from a `fit()` call.
"""
struct FitResult
    model::CompiledModel
    converged::Bool
    ofv::Float64           # final FOCE OFV (≈ -2 log-likelihood)
    aic::Float64
    bic::Float64

    # Population parameter estimates
    theta::Vector{Float64}
    theta_names::Vector{Symbol}
    omega::Matrix{Float64}
    sigma::Vector{Float64}

    # Uncertainty (empty if covariance step failed)
    covariance_matrix::Matrix{Float64}
    se_theta::Vector{Float64}
    se_omega::Vector{Float64}
    se_sigma::Vector{Float64}

    # Per-subject
    subjects::Vector{SubjectResult}

    n_obs::Int
    n_subjects::Int
    n_parameters::Int
    n_iterations::Int
    interaction::Bool        # true = FOCE-I (eta-epsilon interaction)
    warnings::Vector{String}
end

function Base.show(io::IO, r::FitResult)
    status = r.converged ? "converged" : "NOT CONVERGED"
    println(io, "FitResult [$status]")
    println(io, "  OFV: $(round(r.ofv, digits=3))   AIC: $(round(r.aic, digits=3))   BIC: $(round(r.bic, digits=3))")
    println(io, "  Subjects: $(r.n_subjects)   Observations: $(r.n_obs)   Parameters: $(r.n_parameters)")
    println(io, "  Iterations: $(r.n_iterations)")
    println(io, "\n  THETA:")
    for (name, val, se) in zip(r.theta_names, r.theta, r.se_theta)
        isempty(r.se_theta) ?
            println(io, "    $name = $(round(val, digits=4))") :
            println(io, "    $name = $(round(val, digits=4))  (SE=$(round(se, digits=4)))")
    end
    if !isempty(r.warnings)
        println(io, "\n  Warnings:")
        for w in r.warnings; println(io, "    ! $w"); end
    end
end
