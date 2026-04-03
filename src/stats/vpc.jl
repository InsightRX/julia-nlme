"""
Visual Predictive Check (VPC) computation.

Reference: Karlsson & Holford (2008) PAGE 17, Abstr 1434.
           Bergstrand et al. (2011) AAPS J 13(2):143-51 (pred-correction)

A VPC overlays observed data percentiles against the distribution of the same
percentiles computed across simulation replicates. It is the standard
pharmacometric diagnostic for assessing model adequacy.

## Algorithm

For each time bin:
  1. Compute the lower PI, median, and upper PI of `dv` across simulation
     replicates for each replicate separately → (n_sim × 3) matrix per bin.
  2. Compute the CI bounds across those per-replicate percentiles →
     9 values (lo/med/hi for each of the 3 PI percentiles) per bin.
  3. Compute the observed PI directly from the observed `dv` values per bin.

## Prediction correction (pred_corr = true)

Normalises each observation and simulation by its population prediction
relative to the median prediction in the bin:
  dv_corr = dv × median(pred_in_bin) / pred

This makes subjects with different doses or covariate values comparable
on the same axis. Requires a `pred` column in both `obs` and `sim`.
`sim` DataFrames from `simulate()` already contain `pred`.
"""

using Statistics: quantile, mean, median

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

"""
    VPCResult

Output from [`vpc`](@ref).

Fields:
  - `vpc_stat`:  DataFrame with simulation CI bounds per bin — columns:
                 `bin`, `bin_mid`, `bin_min`, `bin_max`,
                 `pi_lo_lo/med/hi`, `p50_lo/med/hi`, `pi_hi_lo/med/hi`
  - `obs_stat`:  DataFrame with observed percentiles per bin — columns:
                 `bin`, `bin_mid`, `bin_min`, `bin_max`,
                 `obs_pi_lo`, `obs_p50`, `obs_pi_hi`
  - `obs`:       Observed data (normalised columns, with `_bin` assigned)
  - `sim`:       Simulated data (normalised columns, with `_bin` assigned)
  - `bin_edges`: Vector of bin boundary time values
  - `pi`:        Prediction interval tuple, e.g. `(0.05, 0.95)`
  - `ci`:        Confidence interval tuple, e.g. `(0.05, 0.95)`
  - `pred_corr`: Whether prediction correction was applied
"""
struct VPCResult
    vpc_stat::DataFrame
    obs_stat::DataFrame
    obs::DataFrame
    sim::DataFrame
    bin_edges::Vector{Float64}
    pi::Tuple{Float64, Float64}
    ci::Tuple{Float64, Float64}
    pred_corr::Bool
end

function Base.show(io::IO, v::VPCResult)
    n_bins = nrow(v.obs_stat)
    n_sim  = isempty(v.sim) ? 0 : length(unique(v.sim._sim))
    pc_str = v.pred_corr ? ", pred-corrected" : ""
    println(io, "VPCResult: $n_bins bins, $n_sim replicates, " *
                "PI=$(v.pi), CI=$(v.ci)$pc_str")
    println(io, "\n  vpc_stat ($(nrow(v.vpc_stat)) rows × $(ncol(v.vpc_stat)) cols):")
    show(io, first(v.vpc_stat, min(3, nrow(v.vpc_stat))); summary=false)
    println(io, "\n\n  obs_stat ($(nrow(v.obs_stat)) rows × $(ncol(v.obs_stat)) cols):")
    show(io, first(v.obs_stat, min(3, nrow(v.obs_stat))); summary=false)
end

# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

"""
    _make_bin_edges(times, bins, n_bins) → Vector{Float64}

Compute bin edges from observed times.

`bins` can be:
  - `:auto` or `:quantile` — equal-count bins (quantile-based)
  - `:equal`               — equal-width bins
  - A `Vector{<:Real}`     — explicit bin edges (used as-is)

`n_bins` is used for `:auto`/`:equal`/`:quantile`. When `n_bins == 0`,
auto-selects `min(max(3, n ÷ 40), 15)` where `n` is the number of observations.
"""
function _make_bin_edges(times::AbstractVector{<:Real},
                          bins::Union{Symbol, AbstractVector{<:Real}},
                          n_bins::Int)::Vector{Float64}
    if bins isa AbstractVector
        return Float64.(bins)
    end

    nb = n_bins > 0 ? n_bins : min(max(3, length(times) ÷ 40), 15)

    if bins == :equal
        edges = range(minimum(times), maximum(times); length = nb + 1)
        return collect(Float64, edges)
    elseif bins == :auto || bins == :quantile
        probs = range(0.0, 1.0; length = nb + 1)
        edges = quantile(times, probs)
        return Float64.(unique(edges))
    else
        error("Unknown binning method: $bins. Use :auto, :equal, :quantile, " *
              "or a Vector of explicit bin edges.")
    end
end

"""
    _assign_bins(times, bin_edges) → Vector{Int}

Assign each time value to a bin index in 1:n_bins.
Bin k spans [bin_edges[k], bin_edges[k+1]).
The last bin is right-closed: [bin_edges[end-1], bin_edges[end]].
"""
function _assign_bins(times::AbstractVector{<:Real},
                       bin_edges::Vector{Float64})::Vector{Int}
    n_bins = length(bin_edges) - 1
    return [clamp(searchsortedlast(bin_edges, t), 1, n_bins) for t in times]
end

# ---------------------------------------------------------------------------
# Main VPC function
# ---------------------------------------------------------------------------

"""
    vpc(obs, sim; bins=:auto, n_bins=0, pi=(0.05, 0.95), ci=(0.05, 0.95),
        stratify=nothing, pred_corr=false)

Compute Visual Predictive Check statistics from observed and simulated data.

Returns a [`VPCResult`](@ref).

## Arguments

  - `obs`: observed data DataFrame. Required columns (case-insensitive):
    `id`, `time`, `dv`. Required when `pred_corr=true`: `pred`.
  - `sim`: simulated data DataFrame, as returned by [`simulate`](@ref).
    Required columns: `id`, `time`, `dv`, `_sim` (replicate index).
    `pred` is required when `pred_corr=true`.

## Keyword arguments

  - `bins`: binning method — `:auto` (quantile-based, default), `:equal`
    (equal-width), `:quantile`, or a `Vector{<:Real}` of explicit bin edges.
  - `n_bins`: number of bins. When 0 (default), auto-selects
    `min(max(3, n_obs ÷ 40), 15)`.
  - `pi`: prediction interval as a 2-tuple, default `(0.05, 0.95)`.
    Controls which percentiles of simulated DV are reported.
  - `ci`: confidence interval around each PI percentile, default `(0.05, 0.95)`.
    Controls the width of the shaded CI ribbons.
  - `stratify`: a `Symbol` or `Vector{Symbol}` of column names to stratify
    by (must exist in both `obs` and `sim`). When set, all statistics are
    computed separately per stratum.
  - `pred_corr`: apply prediction correction (default `false`).
    Normalises DV values by `pred` relative to the per-bin median pred.
    Requires `pred` in both `obs` and `sim`.
"""
function vpc(obs_df::DataFrame,
              sim_df::DataFrame;
              bins::Union{Symbol, AbstractVector{<:Real}} = :auto,
              n_bins::Int                                 = 0,
              pi::Tuple{Float64, Float64}                 = (0.05, 0.95),
              ci::Tuple{Float64, Float64}                 = (0.05, 0.95),
              stratify::Union{Nothing, Symbol, Vector{Symbol}} = nothing,
              pred_corr::Bool                             = false)::VPCResult

    # ---- Normalize column names ----
    obs = copy(obs_df); _normalise_cols(obs)
    sim = copy(sim_df); _normalise_cols(sim)

    for col in (:id, :time, :dv)
        col in propertynames(obs) || error("obs is missing required column: $col")
        col in propertynames(sim) || error("sim is missing required column: $col")
    end

    # Filter to observation records only (drop dose rows, MDV=1, missing DV)
    for (col, default) in [(:evid, 0), (:mdv, 0)]
        col in propertynames(obs) || (obs[!, col] .= default)
    end
    obs_mask = (obs.evid .== 0) .&
               (coalesce.(obs.mdv, 0) .== 0) .&
               (.!ismissing.(obs.dv))
    obs = obs[obs_mask, :]
    :_sim in propertynames(sim) ||
        error("sim is missing the '_sim' replicate-index column. " *
              "Use simulate() to generate the simulated DataFrame.")

    if pred_corr
        :pred in propertynames(obs) ||
            error("pred_corr=true requires a 'pred' column in obs")
        :pred in propertynames(sim) ||
            error("pred_corr=true requires a 'pred' column in sim " *
                  "(simulate() adds this automatically)")
    end

    # ---- Stratification: add a :_strat column ----
    # Normalize stratify column names to match the lowercased DataFrame columns
    strat_cols = stratify === nothing ? Symbol[] :
                 stratify isa Symbol  ? [Symbol(lowercase(string(stratify)))] :
                                        Symbol[Symbol(lowercase(string(s))) for s in stratify]
    if !isempty(strat_cols)
        for col in strat_cols
            col in propertynames(obs) ||
                error("Stratification column '$col' not found in obs")
            col in propertynames(sim) ||
                error("Stratification column '$col' not found in sim")
        end
        # Combine strat columns into a single string key
        obs[!, :_strat] = [join([row[c] for c in strat_cols], "|") for row in eachrow(obs)]
        sim[!, :_strat] = [join([row[c] for c in strat_cols], "|") for row in eachrow(sim)]
    else
        obs[!, :_strat] .= ""
        sim[!, :_strat] .= ""
    end

    # ---- Determine bin edges from observed times (shared across strata) ----
    bin_edges = _make_bin_edges(Float64.(obs.time), bins, n_bins)
    obs[!, :_bin] = _assign_bins(Float64.(obs.time), bin_edges)
    sim[!, :_bin] = _assign_bins(Float64.(sim.time), bin_edges)
    n_bins_actual = length(bin_edges) - 1

    # ---- Prediction correction ----
    if pred_corr
        # Compute median pred per (strat, bin) from observed data,
        # then correct both obs and sim.
        pred_medians = Dict{Tuple{String,Int}, Float64}()
        for grp in groupby(obs, [:_strat, :_bin])
            key = (grp._strat[1], grp._bin[1])
            pred_medians[key] = median(Float64.(grp.pred))
        end

        obs[!, :dv] = [
            let key = (obs._strat[i], obs._bin[i])
                pm = get(pred_medians, key, NaN)
                Float64(obs.dv[i]) * pm / max(Float64(obs.pred[i]), 1e-12)
            end
            for i in 1:nrow(obs)
        ]
        sim[!, :dv] = [
            let key = (sim._strat[i], sim._bin[i])
                pm = get(pred_medians, key, NaN)
                Float64(sim.dv[i]) * pm / max(Float64(sim.pred[i]), 1e-12)
            end
            for i in 1:nrow(sim)
        ]
    end

    # ---- Compute per-(strat, bin) observed percentiles ----
    obs_stat_rows = NamedTuple[]
    for grp in groupby(obs, [:_strat, :_bin])
        b      = grp._bin[1]
        strat  = grp._strat[1]
        y      = Float64.(grp.dv)
        length(y) < 2 && continue

        bin_min = bin_edges[b]
        bin_max = bin_edges[b + 1]
        bin_mid = mean(Float64.(grp.time))

        push!(obs_stat_rows, (
            strat     = strat,
            bin       = b,
            bin_mid   = bin_mid,
            bin_min   = bin_min,
            bin_max   = bin_max,
            obs_pi_lo = quantile(y, pi[1]),
            obs_p50   = quantile(y, 0.5),
            obs_pi_hi = quantile(y, pi[2]),
        ))
    end
    obs_stat = isempty(obs_stat_rows) ? DataFrame() : DataFrame(obs_stat_rows)

    # ---- Compute per-(strat, _sim, bin) simulation percentiles ----
    sim_rep_rows = NamedTuple[]
    for grp in groupby(sim, [:_strat, :_sim, :_bin])
        b      = grp._bin[1]
        strat  = grp._strat[1]
        y      = Float64.(grp.dv)
        length(y) < 1 && continue

        push!(sim_rep_rows, (
            strat = strat,
            _sim  = grp._sim[1],
            _bin  = b,
            q_lo  = quantile(y, pi[1]),
            q_med = quantile(y, 0.5),
            q_hi  = quantile(y, pi[2]),
        ))
    end
    sim_rep = isempty(sim_rep_rows) ? DataFrame() : DataFrame(sim_rep_rows)

    # ---- Compute CI across replicates per (strat, bin) ----
    vpc_stat_rows = NamedTuple[]
    if !isempty(sim_rep)
        for grp in groupby(sim_rep, [:strat, :_bin])
            b     = grp._bin[1]
            strat = grp.strat[1]

            # bin_mid from obs in this (strat, bin)
            obs_b = obs[(obs._strat .== strat) .& (obs._bin .== b), :]
            isempty(obs_b) && continue

            bin_mid = mean(Float64.(obs_b.time))
            bin_min = bin_edges[b]
            bin_max = bin_edges[b + 1]

            q_lo  = Float64.(grp.q_lo)
            q_med = Float64.(grp.q_med)
            q_hi  = Float64.(grp.q_hi)

            push!(vpc_stat_rows, (
                strat      = strat,
                bin        = b,
                bin_mid    = bin_mid,
                bin_min    = bin_min,
                bin_max    = bin_max,
                pi_lo_lo   = quantile(q_lo,  ci[1]),
                pi_lo_med  = quantile(q_lo,  0.5),
                pi_lo_hi   = quantile(q_lo,  ci[2]),
                p50_lo     = quantile(q_med, ci[1]),
                p50_med    = quantile(q_med, 0.5),
                p50_hi     = quantile(q_med, ci[2]),
                pi_hi_lo   = quantile(q_hi,  ci[1]),
                pi_hi_med  = quantile(q_hi,  0.5),
                pi_hi_hi   = quantile(q_hi,  ci[2]),
            ))
        end
    end
    vpc_stat = isempty(vpc_stat_rows) ? DataFrame() : DataFrame(vpc_stat_rows)

    # Tidy up the internal :_strat column
    if isempty(strat_cols)
        # No stratification: drop it entirely from all outputs
        isempty(obs_stat) || select!(obs_stat, Not(:strat))
        isempty(vpc_stat) || select!(vpc_stat, Not(:strat))
        select!(obs, Not(:_strat))
        select!(sim, Not(:_strat))
    else
        # Rename :_strat → :strat in obs/sim so all DataFrames use the same name
        rename!(obs, :_strat => :strat)
        rename!(sim, :_strat => :strat)
    end

    # Sort output by bin
    isempty(obs_stat) || sort!(obs_stat, :bin)
    isempty(vpc_stat) || sort!(vpc_stat, :bin)

    return VPCResult(vpc_stat, obs_stat, obs, sim, bin_edges, pi, ci, pred_corr)
end
