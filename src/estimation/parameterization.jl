"""
Unconstrained parameterization for outer optimization.

To allow gradient-based optimizers to work without box constraints:
  - θ (THETA):  log-transformed positive parameters → unconstrained
  - Ω (OMEGA):  log of diagonal Cholesky elements + raw off-diagonals
  - σ (SIGMA):  log-transformed positive variance parameters

All functions support round-tripping:
  unpack(pack(params)) ≈ params
"""

using LinearAlgebra

# ---------------------------------------------------------------------------
# Pack ModelParameters into a flat unconstrained vector
# ---------------------------------------------------------------------------

"""
    pack_params(params)

Convert `ModelParameters` to a flat `Vector{Float64}` for the optimizer.

Layout: [log(θ)..., chol_params(Ω)..., log(σ)...]

For Ω Cholesky factor L (lower triangular):
  - Diagonal elements: stored as log(Lᵢᵢ)  (ensures Lᵢᵢ > 0)
  - Off-diagonal:      stored as-is (no constraint)
"""
function pack_params(params::ModelParameters)::Vector{Float64}
    v = Float64[]

    # θ: log-transform (assumes θ > 0; if θ can be negative, no transform needed)
    append!(v, log.(params.theta))

    # Ω Cholesky: diagonal-only or full lower triangle
    L = params.omega.chol
    n = size(L, 1)
    for j in 1:n, i in j:n
        params.omega.diagonal && i != j && continue
        push!(v, i == j ? log(L[i, j]) : L[i, j])
    end

    # σ: log-transform
    append!(v, log.(params.sigma.values))

    return v
end

"""
    unpack_params(v, template)

Reconstruct a `ModelParameters` from a flat vector `v`, using `template`
for structural information (names, sizes).
"""
function unpack_params(v::AbstractVector{T},
                        template::ModelParameters) where T<:Real
    idx = 1

    # θ
    n_theta = length(template.theta)
    theta = exp.(v[idx:idx+n_theta-1])
    idx  += n_theta

    # Ω from Cholesky elements
    n_eta = n_etas(template.omega)
    L_raw = zeros(T, n_eta, n_eta)
    for j in 1:n_eta, i in j:n_eta
        if template.omega.diagonal && i != j
            # off-diagonal fixed at zero
            continue
        end
        L_raw[i, j] = i == j ? exp(v[idx]) : v[idx]
        idx += 1
    end
    L   = LowerTriangular(L_raw)
    mat = collect(L * L')
    omega = OmegaMatrix(mat, template.omega.eta_names; diagonal = template.omega.diagonal)

    # σ
    n_sigma = length(template.sigma.values)
    sigma_vals = exp.(v[idx:idx+n_sigma-1])
    sigma = SigmaMatrix(collect(sigma_vals), template.sigma.names)

    return ModelParameters(collect(theta), template.theta_names, omega, sigma)
end

"""
    _unpack_raw(v, template)

Like `unpack_params` but returns raw arrays `(theta, omega_mat, sigma_vals)`
without constructing a `ModelParameters`. This allows ForwardDiff Dual numbers
to flow through when differentiating the FOCE OFV w.r.t. the packed vector.
"""
function _unpack_raw(v::AbstractVector{T}, template::ModelParameters) where T<:Real
    idx = 1

    n_theta = length(template.theta)
    theta   = exp.(v[idx:idx+n_theta-1])
    idx    += n_theta

    n_eta   = n_etas(template.omega)
    L_raw   = zeros(T, n_eta, n_eta)
    for j in 1:n_eta, i in j:n_eta
        if template.omega.diagonal && i != j
            continue
        end
        L_raw[i, j] = i == j ? exp(v[idx]) : v[idx]
        idx += 1
    end
    L         = LowerTriangular(L_raw)
    omega_mat = L * L'

    n_sigma    = length(template.sigma.values)
    sigma_vals = exp.(v[idx:idx+n_sigma-1])

    return theta, omega_mat, sigma_vals
end

"""
    n_packed(params)

Number of elements in the packed parameter vector.
"""
function n_packed(params::ModelParameters)::Int
    n_eta    = n_etas(params.omega)
    n_chol   = params.omega.diagonal ? n_eta : n_eta * (n_eta + 1) ÷ 2
    return length(params.theta) + n_chol + length(params.sigma.values)
end

# ---------------------------------------------------------------------------
# Initial packed vector and bounds
# ---------------------------------------------------------------------------

"""
    initial_packed(params)

Return `(x0, lower, upper)` for the optimizer.

Theta bounds come from `params.theta_lower/upper` (set from the model file's
`[parameters]` block). These prevent the optimizer from reaching degenerate
parameter regions (e.g. CL→0, Q→∞) that are local minima of the FOCE OFV
for sparse data. Omega and sigma get conservative defaults.
"""
function initial_packed(params::ModelParameters)
    x0 = pack_params(params)

    n_eta   = n_etas(params.omega)
    n_chol  = params.omega.diagonal ? n_eta : n_eta * (n_eta + 1) ÷ 2
    n_sigma = length(params.sigma.values)

    # Theta: log-transformed → bounds are log(theta_lower/upper)
    lower_theta = log.(max.(params.theta_lower, 1e-10))
    upper_theta = log.(min.(params.theta_upper, 1e9))

    # Omega Cholesky elements: diagonal stored as log(L_ii), off-diagonal unconstrained.
    # Lower bound -4.0 → L_ii ≥ exp(-4) → ω_ii ≥ exp(-8) ≈ 3e-4.
    # This blocks the log|Ω|→-∞ degenerate minimum while allowing any real PK
    # variability (NONMEM run18 omega values 0.07–0.57 are far above this floor).
    lower_omega = fill(-4.0, n_chol)
    upper_omega = fill(3.0,  n_chol)

    # Sigma: log-transformed; allow from exp(-8)≈3e-4 to exp(5)≈148
    lower_sigma = fill(-8.0, n_sigma)
    upper_sigma = fill(5.0,  n_sigma)

    lower = vcat(lower_theta, lower_omega, lower_sigma)
    upper = vcat(upper_theta, upper_omega, upper_sigma)

    # Clamp initial point to bounds. This is needed when x0 comes from a
    # warm-start (e.g. ITS → FOCE) where estimates may lie outside the FOCE
    # box constraints (e.g. ITS omega collapse → packed value < lower_omega).
    x0 = clamp.(x0, lower, upper)

    return x0, lower, upper
end
