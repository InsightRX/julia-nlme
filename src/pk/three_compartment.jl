"""
Analytical solutions for three-compartment PK models.

Parameterization: CL, V1, Q2, V2, Q3, V3 (two inter-compartmental clearances).
Macro-rate constants α > β > γ > 0 are roots of a cubic equation solved via
the trigonometric (Vieta) method — fully ForwardDiff-compatible.

All functions are parametric in T<:Real for ForwardDiff compatibility.
"""

# ---------------------------------------------------------------------------
# Macro-rate constants from CL/V parameterization
# ---------------------------------------------------------------------------

"""
    _three_cpt_macro_rates(cl, v1, q2, v2, q3, v3)

Compute macro-rate constants α > β > γ > 0 and micro-constants from
the CL/V parameterization.

Micro-constants:
  k10 = CL/V1,  k12 = Q2/V1,  k21 = Q2/V2,  k13 = Q3/V1,  k31 = Q3/V3

Characteristic polynomial for the positive macro-rate constants:
  λ³ - S₂λ² + S₁λ - S₀ = 0
  S₂ = k10+k12+k13+k21+k31
  S₁ = k10·k21 + k10·k31 + k21·k31 + k12·k31 + k13·k21
  S₀ = k10·k21·k31

Solved via the trigonometric method (three distinct real roots guaranteed for
any physiologically valid parameter set).
"""
@inline function _three_cpt_macro_rates(cl::C, v1::V1, q2::Q2, v2::V2,
                                         q3::Q3, v3::V3) where {C<:Real,V1<:Real,Q2<:Real,V2<:Real,Q3<:Real,V3<:Real}
    T = promote_type(C, V1, Q2, V2, Q3, V3)
    cl, v1, q2, v2, q3, v3 = T(cl), T(v1), T(q2), T(v2), T(q3), T(v3)

    k10 = cl / v1
    k12 = q2 / v1
    k21 = q2 / v2
    k13 = q3 / v1
    k31 = q3 / v3

    # Symmetric functions of the roots (Vieta's formulas for the characteristic poly)
    S2 = k10 + k12 + k13 + k21 + k31
    S1 = k10*k21 + k10*k31 + k21*k31 + k12*k31 + k13*k21
    S0 = k10 * k21 * k31

    # Depress the cubic λ = x + S2/3
    h = S2 / 3
    p = S1 - S2^2 / 3               # coefficient of x; p < 0 for 3 real roots
    q = S1*S2/3 - 2*S2^3/27 - S0    # constant term

    # Trigonometric solution for three real roots (p < 0 guaranteed for valid PK params)
    # xₖ = m·cos(φ - 2πk/3),  m = 2√(-p/3),  φ = arccos(3q/(p·m)) / 3
    p_safe = min(p, -eps(T))         # guard against p=0 in degenerate cases
    m   = 2 * sqrt(-p_safe / 3)
    arg = clamp(3*q / (p_safe * m), -one(T), one(T))
    φ   = acos(arg) / 3

    λ0 = m * cos(φ)             + h
    λ1 = m * cos(φ - T(2π/3))  + h
    λ2 = m * cos(φ - T(4π/3))  + h

    # Sort descending: α > β > γ  (avoid if/else by using max/min + Vieta sum)
    α = max(λ0, max(λ1, λ2))
    γ = min(λ0, min(λ1, λ2))
    β = S2 - α - γ

    return α, β, γ, k10, k12, k21, k13, k31
end

# ---------------------------------------------------------------------------
# 3-CMT IV bolus
# ---------------------------------------------------------------------------

"""
    three_cpt_iv_bolus(; cl, v1, q2, v2, q3, v3, dose, t)

Three-compartment IV bolus concentration at time `t`.

C(t) = A·exp(-α·t) + B·exp(-β·t) + G·exp(-γ·t)

Coefficients from residue theorem (partial fractions of the Laplace transform):
  A = (D/V1)·(α-k21)(α-k31) / [(α-β)(α-γ)]
  B = (D/V1)·(β-k21)(β-k31) / [(β-α)(β-γ)]
  G = (D/V1)·(γ-k21)(γ-k31) / [(γ-α)(γ-β)]
"""
function three_cpt_iv_bolus(; cl::C, v1::V1, q2::Q2, v2::V2, q3::Q3, v3::V3,
                               dose::Real, t::TT) where {C<:Real,V1<:Real,Q2<:Real,V2<:Real,Q3<:Real,V3<:Real,TT<:Real}
    α, β, γ, k10, k12, k21, k13, k31 = _three_cpt_macro_rates(cl, v1, q2, v2, q3, v3)
    T   = eltype(α)
    D   = T(dose) / T(v1)
    t_  = T(t)
    αβ  = α - β;  αγ = α - γ;  βγ = β - γ

    A = D * (α - k21) * (α - k31) / (αβ * αγ)
    B = D * (β - k21) * (β - k31) / (-αβ * βγ)
    G = D * (γ - k21) * (γ - k31) / (αγ * βγ)   # = D*(γ-k21)*(γ-k31) / [(γ-α)(γ-β)]

    return A * exp(-α * t_) + B * exp(-β * t_) + G * exp(-γ * t_)
end

# ---------------------------------------------------------------------------
# 3-CMT constant-rate IV infusion
# ---------------------------------------------------------------------------

"""
    three_cpt_infusion(; cl, v1, q2, v2, q3, v3, dose, duration, t)

Three-compartment constant-rate infusion concentration at time `t`.

During infusion (t ≤ T):
  C = (Rate/V1)·[a·(1-exp(-α·t))/α + b·(1-exp(-β·t))/β + c·(1-exp(-γ·t))/γ]

After infusion (t > T):
  Plateau amplitudes at t=T decay with the same macro-rate constants.
"""
function three_cpt_infusion(; cl::C, v1::V1, q2::Q2, v2::V2, q3::Q3, v3::V3,
                               dose::Real, duration::Real, t::TT) where {C<:Real,V1<:Real,Q2<:Real,V2<:Real,Q3<:Real,V3<:Real,TT<:Real}
    α, β, γ, k10, k12, k21, k13, k31 = _three_cpt_macro_rates(cl, v1, q2, v2, q3, v3)
    T    = eltype(α)
    t_   = T(t)
    T_   = T(duration)
    rate = T(dose) / T_
    RV   = rate / T(v1)
    αβ   = α - β;  αγ = α - γ;  βγ = β - γ

    # Normalized IV bolus coefficients (without Dose/V1 factor)
    a = (α - k21) * (α - k31) / (αβ * αγ)
    b = (β - k21) * (β - k31) / (-αβ * βγ)
    c = (γ - k21) * (γ - k31) / (αγ * βγ)

    if t_ <= T_
        return RV * (a * (one(T) - exp(-α * t_)) / α +
                     b * (one(T) - exp(-β * t_)) / β +
                     c * (one(T) - exp(-γ * t_)) / γ)
    else
        Ap = RV * a * (one(T) - exp(-α * T_)) / α
        Bp = RV * b * (one(T) - exp(-β * T_)) / β
        Gp = RV * c * (one(T) - exp(-γ * T_)) / γ
        τ  = t_ - T_
        return Ap * exp(-α * τ) + Bp * exp(-β * τ) + Gp * exp(-γ * τ)
    end
end

# ---------------------------------------------------------------------------
# 3-CMT oral (first-order absorption)
# ---------------------------------------------------------------------------

"""
    three_cpt_oral(; cl, v1, q2, v2, q3, v3, ka, f, dose, t)

Three-compartment first-order oral absorption concentration at time `t`.

The central-compartment response to an exponential depot input F·D·KA·exp(-KA·t) is:

  C(t) = (F·D·KA/V1)·[a·Φ(α,KA,t) + b·Φ(β,KA,t) + c·Φ(γ,KA,t)]

where Φ(λ,KA,t) = (exp(-λ·t) - exp(-KA·t))/(KA-λ), handled via `_bateman_diff`
which smoothly manages the KA≈λ singularity.
"""
three_cpt_oral(; cl, v1, q2, v2, q3, v3, ka, f=1.0, dose, t) =
    _three_cpt_oral_impl(cl, v1, q2, v2, q3, v3, ka, f, dose, t)

function _three_cpt_oral_impl(cl::C, v1::V1, q2::Q2, v2::V2, q3::Q3, v3::V3,
                                ka::K, f::F, dose::Real, t::TT) where {C<:Real,V1<:Real,Q2<:Real,V2<:Real,Q3<:Real,V3<:Real,K<:Real,F<:Real,TT<:Real}
    α, β, γ, k10, k12, k21, k13, k31 = _three_cpt_macro_rates(cl, v1, q2, v2, q3, v3)
    T    = promote_type(eltype(α), K, F, TT)
    ka_  = T(ka)
    t_   = T(t)
    FD   = T(f) * T(dose)
    coef = FD * ka_ / T(v1)
    αβ   = α - β;  αγ = α - γ;  βγ = β - γ

    # Normalized IV bolus residue coefficients
    a = (α - k21) * (α - k31) / (αβ * αγ)
    b = (β - k21) * (β - k31) / (-αβ * βγ)
    c = (γ - k21) * (γ - k31) / (αγ * βγ)

    return coef * (a * _bateman_diff(α, ka_, t_) +
                   b * _bateman_diff(β, ka_, t_) +
                   c * _bateman_diff(γ, ka_, t_))
end

# ---------------------------------------------------------------------------
# Three-compartment single-dose closure builder (called from make_single_dose_fn)
# ---------------------------------------------------------------------------

function _three_cpt_single_dose_fn(pk_model::Symbol, pk_params::NamedTuple)
    if pk_model == :three_cpt_iv_bolus
        cl = pk_params.cl; v1 = pk_params.v1
        q2 = pk_params.q2; v2 = pk_params.v2
        q3 = pk_params.q3; v3 = pk_params.v3
        return (dose::DoseEvent, τ::Real) ->
            three_cpt_iv_bolus(; cl=cl, v1=v1, q2=q2, v2=v2, q3=q3, v3=v3,
                                  dose=dose.amt, t=τ)

    elseif pk_model == :three_cpt_infusion
        cl = pk_params.cl; v1 = pk_params.v1
        q2 = pk_params.q2; v2 = pk_params.v2
        q3 = pk_params.q3; v3 = pk_params.v3
        return (dose::DoseEvent, τ::Real) ->
            three_cpt_infusion(; cl=cl, v1=v1, q2=q2, v2=v2, q3=q3, v3=v3,
                                  dose=dose.amt, duration=dose.duration, t=τ)

    elseif pk_model == :three_cpt_oral
        cl = pk_params.cl; v1 = pk_params.v1
        q2 = pk_params.q2; v2 = pk_params.v2
        q3 = pk_params.q3; v3 = pk_params.v3
        ka = pk_params.ka; f  = get(pk_params, :f, one(cl))
        return (dose::DoseEvent, τ::Real) ->
            _three_cpt_oral_impl(cl, v1, q2, v2, q3, v3, ka, f, dose.amt, τ)
    else
        error("Unknown three-compartment model: $pk_model")
    end
end
