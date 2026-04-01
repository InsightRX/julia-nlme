"""
Analytical solutions for one-compartment PK models.

All functions are parametric in T<:Real to support ForwardDiff dual numbers.
Concentrations are in the central compartment (cmt=2 for oral, cmt=1 for IV).
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""
Safe version of (exp(-a*t) - exp(-b*t)) / (b - a) that handles a ≈ b.
Used in absorption equations. AD-compatible via smooth `ifelse`.
"""
@inline function _bateman_diff(a::A, b::B, t::TT) where {A<:Real, B<:Real, TT<:Real}
    T = promote_type(A, B, TT)
    a, b, t = T(a), T(b), T(t)
    Δ = b - a
    # When |Δ| < ε, L'Hôpital gives t·exp(-a·t)
    # Use smooth blending via ifelse so ForwardDiff can pass through
    small = abs(Δ) < 1e-6
    safe_Δ = ifelse(small, one(T), Δ)
    regular = (exp(-a * t) - exp(-b * t)) / safe_Δ
    limit   = t * exp(-a * t)
    return ifelse(small, limit, regular)
end

# ---------------------------------------------------------------------------
# 1-CMT IV bolus
# ---------------------------------------------------------------------------

"""
    one_cpt_iv_bolus(; cl, v, dose, t)

Single-dose, one-compartment IV bolus concentration at time `t` after dose.

C(t) = (Dose/V) · exp(-k·t),  k = CL/V
"""
function one_cpt_iv_bolus(; cl::C, v::V, dose::Real, t::TT) where {C<:Real, V<:Real, TT<:Real}
    T = promote_type(C, V, TT)
    k = T(cl) / T(v)
    return (T(dose) / T(v)) * exp(-k * T(t))
end

# ---------------------------------------------------------------------------
# 1-CMT constant-rate IV infusion
# ---------------------------------------------------------------------------

"""
    one_cpt_infusion(; cl, v, dose, duration, t)

One-compartment constant-rate infusion concentration at time `t`.

During infusion (t ≤ duration):  C = (Rate/CL)·(1 - exp(-k·t))
After infusion  (t > duration):  C = (Rate/CL)·(1 - exp(-k·T))·exp(-k·(t-T))
"""
function one_cpt_infusion(; cl::C, v::V, dose::Real, duration::Real, t::TT) where {C<:Real, V<:Real, TT<:Real}
    T    = promote_type(C, V, TT)
    k    = T(cl) / T(v)
    rate = T(dose) / T(duration)
    css  = rate / T(cl)
    if t <= duration
        return css * (one(T) - exp(-k * T(t)))
    else
        return css * (one(T) - exp(-k * T(duration))) * exp(-k * (T(t) - T(duration)))
    end
end

# ---------------------------------------------------------------------------
# 1-CMT oral (first-order absorption)
# ---------------------------------------------------------------------------

"""
    one_cpt_oral(; cl, v, ka, f, dose, t)

One-compartment first-order oral absorption concentration at time `t`.

C(t) = (F·Dose·KA) / (V·(KA - k)) · [exp(-k·t) - exp(-KA·t)]

Singularity at KA = k is handled via `_bateman_diff`.
"""
one_cpt_oral(; cl, v, ka, f=1.0, dose, t) =
    _one_cpt_oral_impl(cl, v, ka, f, dose, t)

function _one_cpt_oral_impl(cl::C, v::V, ka::K, f::F,
                              dose::Real, t::TT) where {C<:Real, V<:Real, K<:Real, F<:Real, TT<:Real}
    T     = promote_type(C, V, K, F, TT)
    k     = T(cl) / T(v)
    coeff = T(f) * T(dose) * T(ka) / T(v)
    return coeff * _bateman_diff(k, T(ka), T(t))
end

# ---------------------------------------------------------------------------
# Multi-dose superposition (linear PK)
# ---------------------------------------------------------------------------

"""
    predict_concentration(pk_fn, doses, t)

Compute concentration at time `t` by linear superposition over all doses
that occurred at or before `t`.

`pk_fn(dose_event, τ)` computes the single-dose contribution at lag-time τ.
"""
function predict_concentration(pk_fn::Function, doses::Vector{DoseEvent}, t::T) where T<:Real
    conc = zero(T)
    for dose in doses
        dose.time > t && continue
        τ = T(t - dose.time)
        conc += pk_fn(dose, τ)
    end
    return max(conc, zero(T))
end

# ---------------------------------------------------------------------------
# One-compartment single-dose closure builder (called from make_single_dose_fn)
# ---------------------------------------------------------------------------

function _one_cpt_single_dose_fn(pk_model::Symbol, pk_params::NamedTuple)
    # Capture values from pk_params (may be Float64 or Dual during AD).
    # Type promotion to handle mixed types happens inside the called function.
    if pk_model == :one_cpt_iv_bolus
        cl = pk_params.cl; v = pk_params.v
        return (dose::DoseEvent, τ::Real) ->
            one_cpt_iv_bolus(; cl=cl, v=v, dose=dose.amt, t=τ)

    elseif pk_model == :one_cpt_infusion
        cl = pk_params.cl; v = pk_params.v
        return (dose::DoseEvent, τ::Real) ->
            one_cpt_infusion(; cl=cl, v=v, dose=dose.amt, duration=dose.duration, t=τ)

    elseif pk_model == :one_cpt_oral
        cl = pk_params.cl; v = pk_params.v
        ka = pk_params.ka; f = get(pk_params, :f, one(cl))
        return (dose::DoseEvent, τ::Real) ->
            _one_cpt_oral_impl(cl, v, ka, f, dose.amt, τ)
    else
        error("Unknown one-compartment model: $pk_model")
    end
end

"""
Compute predicted concentrations for a subject at all observation times.

Returns a `Vector{T}` (T=Float64 during normal use; Dual during AD).
"""
function predict_subject(pk_model::Symbol, pk_params::NamedTuple,
                         subject::Subject)
    single_dose_fn = make_single_dose_fn(pk_model, pk_params)
    return [predict_concentration(single_dose_fn, subject.doses, t)
            for t in subject.obs_times]
end
