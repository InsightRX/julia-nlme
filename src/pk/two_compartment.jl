"""
Analytical solutions for two-compartment PK models.

All functions use macro-rate constants (α, β) derived from micro-constants.
Parameterization: CL, V1, Q, V2 (clearance / volume).

All functions are parametric in T<:Real for ForwardDiff compatibility.
"""

# ---------------------------------------------------------------------------
# Macro-rate constants from CL/V parameterization
# ---------------------------------------------------------------------------

"""
    _two_cpt_macro_rates(cl, v1, q, v2)

Compute macro-rate constants α > β from CL, V1, Q, V2.

Returns (α, β, k10, k12, k21) where:
  k10 = CL/V1,  k12 = Q/V1,  k21 = Q/V2
  α, β = eigenvalues of the 2×2 micro-constant matrix
"""
@inline function _two_cpt_macro_rates(cl::C, v1::V1, q::Q, v2::V2) where {C<:Real,V1<:Real,Q<:Real,V2<:Real}
    T = promote_type(C, V1, Q, V2)
    cl, v1, q, v2 = T(cl), T(v1), T(q), T(v2)
    k10 = cl / v1
    k12 = q  / v1
    k21 = q  / v2

    s = k10 + k12 + k21
    d = k10 * k21                   # = k10*k21 (always positive)
    disc = sqrt(max(s^2 - 4d, zero(T)))  # guard against tiny negative due to float error

    α = (s + disc) / 2
    β = (s - disc) / 2

    return α, β, k10, k12, k21
end

# ---------------------------------------------------------------------------
# 2-CMT IV bolus
# ---------------------------------------------------------------------------

"""
    two_cpt_iv_bolus(; cl, v1, q, v2, dose, t)

Two-compartment IV bolus concentration at time `t`.

C(t) = A·exp(-α·t) + B·exp(-β·t)

where:
  A = (Dose/V1)·(α - k21) / (α - β)
  B = (Dose/V1)·(k21 - β) / (α - β)
"""
function two_cpt_iv_bolus(; cl::C, v1::V1, q::Q, v2::V2, dose::Real, t::TT) where {C<:Real,V1<:Real,Q<:Real,V2<:Real,TT<:Real}
    α, β, k10, k12, k21 = _two_cpt_macro_rates(cl, v1, q, v2)
    T  = eltype(α)
    D  = T(dose) / T(v1)
    t_ = T(t)
    αβ = α - β
    A  = D * (α - k21) / αβ
    B  = D * (k21 - β) / αβ
    return A * exp(-α * t_) + B * exp(-β * t_)
end

# ---------------------------------------------------------------------------
# 2-CMT constant-rate IV infusion
# ---------------------------------------------------------------------------

"""
    two_cpt_infusion(; cl, v1, q, v2, dose, duration, t)

Two-compartment constant-rate infusion.

During infusion (t ≤ T):  C = (Rate/V1) · [A'·(1-exp(-α·t))/α + B'·(1-exp(-β·t))/β]
After infusion  (t > T):  Superpose bolus solution evaluated at (t - T)
"""
function two_cpt_infusion(; cl::C, v1::V1, q::Q, v2::V2,
                            dose::Real, duration::Real, t::TT) where {C<:Real,V1<:Real,Q<:Real,V2<:Real,TT<:Real}
    α, β, k10, k12, k21 = _two_cpt_macro_rates(cl, v1, q, v2)
    T    = eltype(α)
    t_   = T(t)
    rate = T(dose) / T(duration)
    αβ   = α - β
    Ar   = (rate / T(v1)) * (α - k21) / αβ
    Br   = (rate / T(v1)) * (k21 - β) / αβ

    if t_ <= T(duration)
        return Ar * (one(T) - exp(-α * t_)) / α +
               Br * (one(T) - exp(-β * t_)) / β
    else
        T_inf = T(duration)
        Ap = Ar * (one(T) - exp(-α * T_inf)) / α
        Bp = Br * (one(T) - exp(-β * T_inf)) / β
        τ  = t_ - T_inf
        return Ap * exp(-α * τ) + Bp * exp(-β * τ)
    end
end

# ---------------------------------------------------------------------------
# 2-CMT oral (first-order absorption)
# ---------------------------------------------------------------------------

"""
    two_cpt_oral(; cl, v1, q, v2, ka, f, dose, t)

Two-compartment first-order oral absorption concentration.

C(t) = P·exp(-α·t) + Q_coef·exp(-β·t) + R·exp(-KA·t)

Coefficients derived from the three-compartment (depot+central+peripheral) system.
Singularities at KA ≈ α or KA ≈ β are handled with smooth `_bateman_diff`.
"""
two_cpt_oral(; cl, v1, q, v2, ka, f=1.0, dose, t) =
    _two_cpt_oral_impl(cl, v1, q, v2, ka, f, dose, t)

function _two_cpt_oral_impl(cl::C, v1::V1, q::Q, v2::V2, ka::K,
                              f::F, dose::Real, t::TT) where {C<:Real,V1<:Real,Q<:Real,V2<:Real,K<:Real,F<:Real,TT<:Real}
    α, β, k10, k12, k21 = _two_cpt_macro_rates(cl, v1, q, v2)
    T   = promote_type(eltype(α), K, F, TT)
    ka_ = T(ka)
    t_  = T(t)
    v1_ = T(v1)
    FD  = T(f) * T(dose)

    αβ  = α - β
    αka = α - ka_
    βka = β - ka_

    use_lim_α = abs(αka) < 1e-6
    use_lim_β = abs(βka) < 1e-6
    safe_αka  = ifelse(use_lim_α, one(T), αka)
    safe_βka  = ifelse(use_lim_β, one(T), βka)

    P_std = FD * ka_ * (k21 - α)  / (v1_ * αβ * safe_αka)
    Q_std = FD * ka_ * (k21 - β)  / (v1_ * αβ * (-safe_βka))
    P_lim = FD * ka_ * (k21 - α)  / (v1_ * αβ) * t_
    Q_lim = FD * ka_ * (k21 - β)  / (v1_ * αβ) * (-t_)
    R     = -(FD * ka_ * (k21 - ka_)) / (v1_ * safe_αka * safe_βka)
    R     = ifelse(use_lim_α | use_lim_β, zero(T), R)

    if !use_lim_α && !use_lim_β
        return P_std * exp(-α * t_) + Q_std * exp(-β * t_) + R * exp(-ka_ * t_)
    elseif use_lim_α
        return P_lim * exp(-α * t_) + Q_std * exp(-β * t_)
    else
        return P_std * exp(-α * t_) + Q_lim * exp(-β * t_)
    end
end

# ---------------------------------------------------------------------------
# Two-compartment single-dose closure builder (called from make_single_dose_fn)
# ---------------------------------------------------------------------------

function _two_cpt_single_dose_fn(pk_model::Symbol, pk_params::NamedTuple)
    if pk_model == :two_cpt_iv_bolus
        cl = pk_params.cl; v1 = pk_params.v1
        q  = pk_params.q;  v2 = pk_params.v2
        return (dose::DoseEvent, τ::Real) ->
            two_cpt_iv_bolus(; cl=cl, v1=v1, q=q, v2=v2, dose=dose.amt, t=τ)

    elseif pk_model == :two_cpt_infusion
        cl = pk_params.cl; v1 = pk_params.v1
        q  = pk_params.q;  v2 = pk_params.v2
        return (dose::DoseEvent, τ::Real) ->
            two_cpt_infusion(; cl=cl, v1=v1, q=q, v2=v2,
                               dose=dose.amt, duration=dose.duration, t=τ)

    elseif pk_model == :two_cpt_oral
        cl = pk_params.cl; v1 = pk_params.v1
        q  = pk_params.q;  v2 = pk_params.v2
        ka = pk_params.ka; f  = get(pk_params, :f, one(cl))
        return (dose::DoseEvent, τ::Real) ->
            _two_cpt_oral_impl(cl, v1, q, v2, ka, f, dose.amt, τ)
    else
        error("Unknown two-compartment model: $pk_model")
    end
end
