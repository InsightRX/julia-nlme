"""
JuliaNLME — Non-Linear Mixed Effects modeling for pharmacokinetics.

Provides FOCE estimation for analytical PK models (1- and 2-compartment).

Quick start:
  using JuliaNLME
  result = fit("model.jnlme", "data.csv")
  print_results(result)
"""
module JuliaNLME

using LinearAlgebra
using Random
using Printf
using Statistics: mean
using CSV, DataFrames
using ForwardDiff
using Optim
using LogExpFunctions
using RuntimeGeneratedFunctions
using Tables

RuntimeGeneratedFunctions.init(@__MODULE__)

# Core types (must be loaded first — everything else depends on these)
include("types.jl")

# PK analytical equations
include("pk/one_compartment.jl")
include("pk/two_compartment.jl")

"""
    make_single_dose_fn(pk_model, pk_params)

Build a single-dose concentration closure for a given PK model symbol.
Returns `f(dose::DoseEvent, τ::T) → T` where τ is time since dose.
Dispatches to one- or two-compartment implementations.
"""
function make_single_dose_fn(pk_model::Symbol, pk_params::NamedTuple)
    if pk_model in (:one_cpt_iv_bolus, :one_cpt_infusion, :one_cpt_oral)
        return _one_cpt_single_dose_fn(pk_model, pk_params)
    elseif pk_model in (:two_cpt_iv_bolus, :two_cpt_infusion, :two_cpt_oral)
        return _two_cpt_single_dose_fn(pk_model, pk_params)
    else
        error("Unknown PK model: $pk_model. " *
              "Supported: :one_cpt_iv_bolus, :one_cpt_infusion, :one_cpt_oral, " *
              ":two_cpt_iv_bolus, :two_cpt_infusion, :two_cpt_oral")
    end
end

# Statistics
include("stats/residual_error.jl")
include("stats/likelihood.jl")

# Estimation
include("estimation/parameterization.jl")
include("estimation/inner_optimizer.jl")
include("estimation/outer_optimizer.jl")
include("estimation/saem.jl")

# IO
include("io/datareader.jl")
include("io/output.jl")

# Model DSL parser
include("parser/model_parser.jl")

# Public API
include("api.jl")

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export fit, fit_saem, simulate
export read_data
export parse_model_file, parse_model_string
export print_results, parameter_table, sdtab
export residual_variance, compute_R_diag, iwres, cwres
export pack_params, unpack_params

# Types
export Population, Subject, DoseEvent
export ModelParameters, OmegaMatrix, SigmaMatrix, n_etas
export CompiledModel, FitResult, SubjectResult

# PK equations (for direct use / testing)
export one_cpt_iv_bolus, one_cpt_infusion, one_cpt_oral
export two_cpt_iv_bolus, two_cpt_infusion, two_cpt_oral
export predict_subject

end # module
