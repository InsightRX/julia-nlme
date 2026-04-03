# julianlme R bridge
# Sourced once by jnlme_setup() via JuliaCall::julia_source().
# All functions are called from R via JuliaCall::julia_call().
#
# Data transfer convention:
#   - R data.frames → temp CSV written in R, path passed as String
#   - Julia DataFrames → temp CSV written by Julia, path returned as String, read in R
#   - Scalar results → returned directly (Float64/Bool/Int → numeric/logical/integer in R)
#   - Named collections → Dict{String,Any} with homogeneous-typed leaves (R receives as list)

using JuliaNLME, DataFrames, CSV
import Random

# ---------------------------------------------------------------------------
# Object caches — keep Julia objects alive between R calls
# ---------------------------------------------------------------------------

const _model_cache  = Dict{String, JuliaNLME.CompiledModel}()
const _result_cache = Dict{String, JuliaNLME.FitResult}()

_newkey() = string(rand(Random.RandomDevice(), UInt64), base = 16)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

function r_parse_model(path::String)::String
    model = JuliaNLME.parse_model_file(path)
    key   = _newkey()
    _model_cache[key] = model
    return key
end

function r_model_info(model_key::String)::Dict{String, Any}
    m  = _model_cache[model_key]
    p  = m.default_params
    ne = JuliaNLME.n_etas(p.omega)
    return Dict{String, Any}(
        "name"        => m.name,
        "pk_model"    => String(m.pk_model),
        "error_model" => String(m.error_model),
        "n_theta"     => m.n_theta,
        "n_eta"       => ne,
        "theta_names" => String.(p.theta_names),
        "eta_names"   => String.(p.omega.eta_names),
        "sigma_names" => String.(p.sigma.names),
        "theta_init"  => collect(Float64, p.theta),
        "theta_lower" => collect(Float64, p.theta_lower),
        "theta_upper" => collect(Float64, p.theta_upper),
        "omega_init"  => vec(collect(Float64, p.omega.matrix)),  # column-major flat
        "sigma_init"  => collect(Float64, p.sigma.values),
    )
end

function r_clear_cache()
    empty!(_model_cache)
    empty!(_result_cache)
    return nothing
end

# ---------------------------------------------------------------------------
# Result serialisation helpers
# ---------------------------------------------------------------------------

function _write_diagnostics(subjects::Vector{JuliaNLME.SubjectResult},
                              eta_names::Vector{Symbol})

    # Long-format: one row per observation per subject
    diag_rows = NamedTuple[]
    for s in subjects
        n = length(s.ipred)
        for j in 1:n
            push!(diag_rows, (
                id    = s.id,
                ipred = s.ipred[j],
                pred  = s.pred[j],
                iwres = s.iwres[j],
                cwres = s.cwres[j],
            ))
        end
    end

    # Wide ETA table: one row per subject
    ids      = [s.id for s in subjects]
    eta_cols = [Symbol("eta_$(String(nm))") => [s.eta[k] for s in subjects]
                for (k, nm) in enumerate(eta_names)]
    eta_df   = DataFrame(vcat([:id => ids], eta_cols))

    diag_path = tempname() * ".csv"
    eta_path  = tempname() * ".csv"
    CSV.write(diag_path, DataFrame(diag_rows))
    CSV.write(eta_path,  eta_df)
    return diag_path, eta_path
end

function _result_to_dict(result::JuliaNLME.FitResult, key::String)::Dict{String, Any}
    eta_names  = result.model.default_params.omega.eta_names
    diag_path, eta_path = _write_diagnostics(result.subjects, eta_names)
    n_om = size(result.omega, 1)

    return Dict{String, Any}(
        "_handle"         => key,
        "converged"       => result.converged,
        "ofv"             => result.ofv,
        "aic"             => result.aic,
        "bic"             => result.bic,
        # Parameters
        "theta"           => collect(Float64, result.theta),
        "theta_names"     => String.(result.theta_names),
        "omega"           => vec(collect(Float64, result.omega)),  # flat column-major
        "omega_dim"       => n_om,
        "eta_names"       => String.(eta_names),
        "sigma"           => collect(Float64, result.sigma),
        "sigma_names"     => String.(result.model.default_params.sigma.names),
        # Standard errors (empty vectors if covariance step failed)
        "se_theta"        => collect(Float64, result.se_theta),
        "se_omega"        => collect(Float64, result.se_omega),
        "se_sigma"        => collect(Float64, result.se_sigma),
        # Metadata
        "n_obs"           => result.n_obs,
        "n_subjects"      => result.n_subjects,
        "n_parameters"    => result.n_parameters,
        "n_iterations"    => result.n_iterations,
        "interaction"     => result.interaction,
        "warnings"        => collect(String, result.warnings),
        # Paths for R to read back
        "diagnostics_csv" => diag_path,
        "eta_csv"         => eta_path,
    )
end

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

function r_fit(model_key::String, data_csv::String;
               interaction::Bool         = false,
               outer_maxiter::Int        = 500,
               outer_gtol::Float64       = 1e-6,
               inner_maxiter::Int        = 200,
               run_covariance_step::Bool = true,
               nthreads::Int             = 1,
               verbose::Bool             = true)::Dict{String, Any}

    model  = _model_cache[model_key]
    df     = CSV.read(data_csv, DataFrame)
    result = JuliaNLME.fit(model, df;
                            interaction, outer_maxiter, outer_gtol,
                            inner_maxiter, run_covariance_step,
                            nthreads, verbose)
    key = _newkey()
    _result_cache[key] = result
    return _result_to_dict(result, key)
end

function r_fit_its(model_key::String, data_csv::String;
                   n_iter::Int               = 100,
                   conv_window::Int          = 20,
                   rel_tol::Float64          = 1e-4,
                   run_covariance_step::Bool = true,
                   interaction::Bool         = false,
                   nthreads::Int             = 1,
                   verbose::Bool             = true)::Dict{String, Any}

    model  = _model_cache[model_key]
    df     = CSV.read(data_csv, DataFrame)
    result = JuliaNLME.fit_its(model, df;
                                n_iter, conv_window, rel_tol,
                                run_covariance_step, interaction,
                                nthreads, verbose)
    key = _newkey()
    _result_cache[key] = result
    return _result_to_dict(result, key)
end

function r_fit_saem(model_key::String, data_csv::String;
                    n_iter_exploration::Int   = 150,
                    n_iter_convergence::Int   = 250,
                    n_mh_steps::Int           = 2,
                    run_covariance_step::Bool = true,
                    interaction::Bool         = false,
                    nthreads::Int             = 1,
                    verbose::Bool             = true)::Dict{String, Any}

    model  = _model_cache[model_key]
    df     = CSV.read(data_csv, DataFrame)
    result = JuliaNLME.fit_saem(model, df;
                                  n_iter_exploration, n_iter_convergence,
                                  n_mh_steps, run_covariance_step,
                                  interaction, nthreads, verbose)
    key = _newkey()
    _result_cache[key] = result
    return _result_to_dict(result, key)
end

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

function r_simulate(model_key::String, result_key::String, data_csv::String;
                    n_sims::Int = 1)::String
    model  = _model_cache[model_key]
    result = _result_cache[result_key]
    df     = CSV.read(data_csv, DataFrame)
    sim_df = JuliaNLME.simulate(model, result, df; n_sims)
    out    = tempname() * ".csv"
    CSV.write(out, sim_df)
    return out
end
