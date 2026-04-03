"""
ODE-based prediction engine for JuliaNLME.

Solves the system of ODEs defined in the model's `[odes]` block to compute
individual predictions at observation times. Handles bolus dose events as
state discontinuities between integration segments.

Performance design:
  - The parser generates out-of-place ODE functions returning `SVector`.
  - `_ode_predictions` dispatches on `Val(N)` (N = number of states) so the
    compiler sees a concrete `SVector{N, T}` type throughout the solve.
  - This lets OrdinaryDiffEq use its static-array code path: no heap
    allocations during integration steps, even when T is a ForwardDiff Dual.
  - Looser default tolerances (1e-6 / 1e-4) are appropriate for PK problems
    and give 2–5× faster integration vs. the default 1e-8 / 1e-6.
"""

using OrdinaryDiffEq
using StaticArrays

# ---------------------------------------------------------------------------
# Public entry point — dispatches to statically-typed implementation
# ---------------------------------------------------------------------------

"""
    _ode_predictions(ode_spec, pk_params, subject)

Integrate the ODE system for one subject and return predicted concentrations
at all observation times.

Dose events are handled by breaking the timeline at each dose time, applying
the instantaneous bolus to the appropriate state, and continuing from the
updated state. Only bolus doses (`rate == 0`) are currently supported.
"""
function _ode_predictions(ode_spec::ODESpec,
                           pk_params::NamedTuple,
                           subject::Subject)
    T = typeof(first(values(pk_params)))   # Float64 or ForwardDiff.Dual
    return _ode_pred_sv(Val(length(ode_spec.state_names)), ode_spec, pk_params, subject, T)
end

# ---------------------------------------------------------------------------
# Statically-dispatched implementation — N known at compile time
# ---------------------------------------------------------------------------

function _ode_pred_sv(::Val{N}, ode_spec::ODESpec, pk_params::NamedTuple,
                       subject::Subject, ::Type{T}) where {N, T}

    # SVector initial condition: zero partials, stack-allocated, no heap use
    u       = zero(SVector{N, T})
    n_obs   = length(subject.obs_times)
    # NaN-fill so that partially-failed ODE integrations (e.g. stiff system
    # causing Tsit5 NaN-dt exit) produce detectable non-finite predictions
    # rather than uninitialized garbage.
    obs_out = fill(T(NaN), n_obs)

    obs_time_idx = Dict{Float64, Int}(t => i for (i, t) in enumerate(subject.obs_times))

    # Break the timeline at dose times to handle state discontinuities
    dose_times  = Float64[d.time for d in subject.doses]
    t_last      = maximum(subject.obs_times)
    break_times = sort(unique(Float64[0.0; dose_times; t_last]))

    for k in 1:(length(break_times) - 1)
        t_start = break_times[k]
        t_end   = break_times[k + 1]

        # Apply instantaneous bolus doses at t_start (SVector is immutable)
        for dose in subject.doses
            if dose.time == t_start
                dose.rate == 0.0 ||
                    error("Infusion doses (rate > 0) are not yet supported in ODE models")
                u_mut = MVector(u)
                u_mut[dose.cmt] += dose.amt   # Float64 amt promotes to T
                u = SVector(u_mut)
            end
        end

        # Record observations exactly at t_start (after dose applied)
        if haskey(obs_time_idx, t_start)
            obs_out[obs_time_idx[t_start]] = u[ode_spec.obs_cmt_idx]
        end

        # Observation times in this segment; always include t_end so
        # sol.u[end] gives the correct initial condition for the next segment.
        seg_obs = filter(t -> t_start < t <= t_end, subject.obs_times)
        saveat  = sort(unique(Float64[seg_obs; t_end]))

        prob = ODEProblem(ode_spec.ode_fn, u, (t_start, t_end), pk_params)
        sol  = solve(prob, Tsit5();
                     saveat  = saveat,
                     dense   = false,
                     abstol  = 1e-6,
                     reltol  = 1e-4)

        # Extract concentrations at observation times in this segment
        for (sol_t, sol_u) in zip(sol.t, sol.u)
            if haskey(obs_time_idx, sol_t)
                obs_out[obs_time_idx[sol_t]] = sol_u[ode_spec.obs_cmt_idx]
            end
        end

        u = sol.u[end]   # SVector; immutable, no copy needed
    end

    return obs_out
end
