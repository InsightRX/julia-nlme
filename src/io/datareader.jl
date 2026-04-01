"""
NONMEM-format CSV dataset reader.

Expected columns (case-insensitive):
  Required: ID, TIME, DV
  Dose:     AMT, EVID (0=obs, 1=dose, 4=reset+dose), CMT, RATE, MDV, II, SS
  Optional: any additional columns are treated as covariates

EVID conventions:
  0 = observation record
  1 = dose record
  2 = other event (ignored)
  4 = reset dose (clear compartments, then dose)
"""

using CSV, DataFrames

# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------

const REQUIRED_COLS  = [:id, :time, :dv]
const DOSE_COLS      = [:amt, :evid, :cmt, :rate, :mdv, :ii, :ss]
const RESERVED_COLS  = Set(vcat(REQUIRED_COLS, DOSE_COLS))

_normalise_cols(df::DataFrame) =
    rename!(df, [c => Symbol(lowercase(c)) for c in names(df)])

function _check_required(df::DataFrame)
    for col in REQUIRED_COLS
        col in propertynames(df) || error("Dataset missing required column: $col")
    end
end

# ---------------------------------------------------------------------------
# Defaults for optional columns
# ---------------------------------------------------------------------------

_col_or_default(df, col, default) =
    col in propertynames(df) ? df[!, col] : fill(default, nrow(df))

# ---------------------------------------------------------------------------
# Parse a single subject's rows
# ---------------------------------------------------------------------------

function _parse_subject(id::Int, rows::DataFrame,
                         const_cov_names::Vector{Symbol},
                         tv_cov_names::Vector{Symbol})::Subject
    doses    = DoseEvent[]
    obs_times = Float64[]
    obs_vals  = Float64[]
    obs_cmts  = Int[]

    # Time-constant covariates: first non-missing value
    covariates = Dict{Symbol, Float64}()
    for col in const_cov_names
        vals = skipmissing(rows[!, col])
        covariates[col] = isempty(vals) ? 0.0 : Float64(first(vals))
    end

    # Time-varying: accumulate per-observation values with LOCF
    tv_last   = Dict{Symbol, Float64}(col => 0.0 for col in tv_cov_names)
    tv_series = Dict{Symbol, Vector{Float64}}(col => Float64[] for col in tv_cov_names)

    for row in eachrow(rows)
        # Update LOCF state for all TV covariates on every row
        for col in tv_cov_names
            if !ismissing(row[col])
                tv_last[col] = Float64(row[col])
            end
        end

        evid = Int(row.evid)
        time = Float64(row.time)

        if evid == 1 || evid == 4
            amt      = ismissing(row.amt)  ? 0.0   : Float64(row.amt)
            rate_val = ismissing(row.rate) ? 0.0   : Float64(row.rate)
            cmt_val  = ismissing(row.cmt)  ? 1     : Int(row.cmt)
            ii_val   = ismissing(row.ii)   ? 0.0   : Float64(row.ii)
            ss_val   = ismissing(row.ss)   ? false : Bool(row.ss)
            dur      = rate_val > 0 ? amt / rate_val : 0.0
            push!(doses, DoseEvent(time, amt, cmt_val, rate_val, dur, ss_val, ii_val))

        elseif evid == 0
            mdv = ismissing(row.mdv) ? 0 : Int(row.mdv)
            dv  = row.dv
            if mdv == 0 && !ismissing(dv)
                cmt_val = ismissing(row.cmt) ? 1 : Int(row.cmt)
                push!(obs_times, time)
                push!(obs_vals,  Float64(dv))
                push!(obs_cmts,  cmt_val)
                # Record LOCF-carried covariate value at this observation
                for col in tv_cov_names
                    push!(tv_series[col], tv_last[col])
                end
            end
        end
        # evid == 2 and others: skip
    end

    sort!(doses, by = d -> d.time)

    return Subject(id, doses, obs_times, obs_vals, obs_cmts, covariates, tv_series)
end

"""
Classify covariate columns as time-constant or time-varying across the full dataset.
A covariate is time-varying if its value changes within at least one subject.
"""
function _classify_covariates(df::DataFrame, covariate_names::Vector{Symbol})
    tv = Symbol[]
    for col in covariate_names
        col in propertynames(df) || continue
        for (_, grp) in pairs(groupby(df, :id))
            unique_vals = unique(skipmissing(grp[!, col]))
            if length(unique_vals) > 1
                push!(tv, col)
                break
            end
        end
    end
    const_covs = [c for c in covariate_names if !(c in tv)]
    return const_covs, tv
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    read_data(path; dv_column=:dv, covariate_columns=nothing)

Read a NONMEM-format CSV file and return a `Population`.

Arguments:
- `path`: path to CSV file
- `dv_column`: name of the dependent variable column (default `:dv`)
- `covariate_columns`: vector of column name symbols to treat as covariates.
  If `nothing`, all non-reserved columns are used as covariates.
"""
function read_data(path::AbstractString;
                   dv_column::Symbol = :dv,
                   covariate_columns::Union{Nothing, Vector{Symbol}} = nothing)::Population

    df = CSV.read(path, DataFrame; missingstring=[".", "NA", ""])
    _normalise_cols(df)
    _check_required(df)

    # Fill optional columns with defaults
    for (col, default) in [(:evid, 0), (:amt, missing), (:cmt, 1),
                             (:rate, 0.0), (:mdv, 0), (:ii, 0.0), (:ss, 0)]
        if !(col in propertynames(df))
            df[!, col] .= default
        end
    end

    cov_cols = if covariate_columns !== nothing
        covariate_columns
    else
        [c for c in propertynames(df) if !(c in RESERVED_COLS)]
    end

    const_cols, tv_cols = _classify_covariates(df, cov_cols)

    subjects = Subject[]
    for (id_val, group) in pairs(groupby(df, :id))
        id = Int(id_val.id)
        push!(subjects, _parse_subject(id, DataFrame(group), const_cols, tv_cols))
    end

    return Population(subjects, cov_cols, dv_column)
end

"""
    read_data(df::DataFrame; kwargs...)

Accept an already-loaded DataFrame instead of a file path.
"""
function read_data(df::DataFrame;
                   dv_column::Symbol = :dv,
                   covariate_columns::Union{Nothing, Vector{Symbol}} = nothing)::Population
    df2 = copy(df)
    _normalise_cols(df2)
    _check_required(df2)

    for (col, default) in [(:evid, 0), (:amt, missing), (:cmt, 1),
                             (:rate, 0.0), (:mdv, 0), (:ii, 0.0), (:ss, 0)]
        if !(col in propertynames(df2))
            df2[!, col] .= default
        end
    end

    cov_cols = if covariate_columns !== nothing
        covariate_columns
    else
        [c for c in propertynames(df2) if !(c in RESERVED_COLS)]
    end

    const_cols, tv_cols = _classify_covariates(df2, cov_cols)

    subjects = Subject[]
    for (id_val, group) in pairs(groupby(df2, :id))
        id = Int(id_val.id)
        push!(subjects, _parse_subject(id, DataFrame(group), const_cols, tv_cols))
    end

    return Population(subjects, cov_cols, dv_column)
end
