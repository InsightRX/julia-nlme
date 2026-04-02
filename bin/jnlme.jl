"""
JuliaNLME command-line interface.

Usage:
  jnlme fit <model.jnlme> <data.csv> [options]

Config file:
  Place a `.jnlme` TOML file in the working directory to set defaults.
  CLI arguments always override config file values.
"""

using ArgParse
using JuliaNLME
using Printf
using TOML

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

function build_arg_parser()
    s = ArgParseSettings(
        prog        = "jnlme",
        description = "JuliaNLME — Non-Linear Mixed Effects estimation",
        version     = "0.1.0",
        add_version = true,
    )

    @add_arg_table! s begin
        "command"
            help     = "Command: fit"
            required = true
        "model"
            help     = "Path to model file (.jnlme)"
            required = true
        "data"
            help     = "Path to data CSV file"
            required = true

        "--optimizer", "-O"
            help     = "Optimizer: lbfgs (default), bfgs, LD_SLSQP, LD_MMA, LD_TNEWTON_PRECOND_RESTART"
            metavar  = "NAME"
            default  = nothing
        "--iterations", "-n"
            help     = "Maximum outer iterations"
            metavar  = "N"
            arg_type = Int
            default  = nothing
        "--interaction"
            help     = "Use FOCE-I (eta-epsilon interaction, recommended for proportional/combined error)"
            action   = :store_true
        "--no-covariance"
            help     = "Skip the covariance step (faster, no standard errors)"
            action   = :store_true
        "--n-starts"
            help     = "Number of multi-start runs (Latin Hypercube Sampling)"
            metavar  = "N"
            arg_type = Int
            default  = nothing
        "--global-search"
            help     = "Enable global pre-search (GN_CRS2_LM) before local optimizer"
            action   = :store_true
        "--covariates"
            help     = "Comma-separated covariate column names (e.g. wt,age,sex). Auto-detected if omitted."
            metavar  = "COLS"
            default  = nothing
        "--output-dir", "-o"
            help     = "Directory for output files (default: current directory)"
            metavar  = "DIR"
            default  = nothing
        "--prefix"
            help     = "Filename prefix for output files (default: model basename)"
            metavar  = "STR"
            default  = nothing
        "--config", "-c"
            help     = "Config file path (default: auto-detect .jnlme in working directory)"
            metavar  = "FILE"
            default  = nothing
        "--quiet", "-q"
            help     = "Suppress iteration progress"
            action   = :store_true
    end

    return s
end

# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

"""
    load_config(path)

Load a TOML config file. Returns a Dict; missing keys get filled with defaults later.
"""
function load_config(path::Union{String, Nothing})
    candidate = if path !== nothing
        path
    elseif isfile(".jnlme")
        ".jnlme"
    else
        nothing
    end

    candidate === nothing && return Dict{String, Any}()

    cfg = TOML.parsefile(candidate)
    @info "Loaded config from $candidate"
    return cfg
end

"""
    cfg_get(cfg, section, key, default)

Look up cfg[section][key] with a default fallback.
"""
cfg_get(cfg, section, key, default) =
    get(get(cfg, section, Dict()), key, default)

# ---------------------------------------------------------------------------
# Merge: CLI args override config, config overrides built-in defaults.
# ---------------------------------------------------------------------------

struct FitOptions
    optimizer           ::Symbol
    outer_maxiter       ::Int
    outer_gtol          ::Float64
    inner_maxiter       ::Int
    inner_tol           ::Float64
    interaction         ::Bool
    run_covariance_step ::Bool
    n_starts            ::Int
    global_search       ::Bool
    verbose             ::Bool
    covariate_columns   ::Union{Nothing, Vector{Symbol}}
    output_dir          ::String
    prefix              ::String
end

function resolve_options(args, cfg)
    e = "estimation"

    optimizer = if args["optimizer"] !== nothing
        Symbol(args["optimizer"])
    else
        Symbol(cfg_get(cfg, e, "optimizer", "lbfgs"))
    end

    outer_maxiter = something(
        args["iterations"],
        cfg_get(cfg, e, "outer_maxiter", nothing),
        500)

    outer_gtol = cfg_get(cfg, e, "outer_gtol", 1e-6)
    inner_maxiter = cfg_get(cfg, e, "inner_maxiter", 200)
    inner_tol = cfg_get(cfg, e, "inner_tol", 1e-8)

    interaction = args["interaction"] ||
                  cfg_get(cfg, e, "interaction", false)

    run_covariance_step = !args["no-covariance"] &&
                          cfg_get(cfg, e, "run_covariance_step", true)

    n_starts = something(
        args["n-starts"],
        cfg_get(cfg, e, "n_starts", nothing),
        1)

    global_search = args["global-search"] ||
                    cfg_get(cfg, e, "global_search", false)

    verbose = !args["quiet"] &&
              cfg_get(cfg, e, "verbose", true)

    # Covariates: CLI → config → nothing (auto-detect)
    covariate_columns = if args["covariates"] !== nothing
        [Symbol(strip(s)) for s in split(args["covariates"], ',')]
    elseif haskey(get(cfg, "data", Dict()), "covariate_columns")
        [Symbol(s) for s in cfg["data"]["covariate_columns"]]
    else
        nothing
    end

    out_section = get(cfg, "output", Dict())
    output_dir = something(args["output-dir"],
                           get(out_section, "directory", nothing),
                           ".")
    prefix_raw = something(args["prefix"],
                           get(out_section, "prefix", nothing),
                           "")
    prefix = isempty(prefix_raw) ?
        splitext(basename(args["model"]))[1] :
        prefix_raw

    return FitOptions(
        optimizer, outer_maxiter, outer_gtol, inner_maxiter, inner_tol,
        interaction, run_covariance_step, n_starts, global_search, verbose,
        covariate_columns, output_dir, prefix,
    )
end

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

function write_params_csv(result::FitResult, path::String)
    open(path, "w") do io
        println(io, "parameter,estimate,se,rse_pct")
        for (i, (name, val)) in enumerate(zip(result.theta_names, result.theta))
            se = isempty(result.se_theta) ? NaN : result.se_theta[i]
            rse = isfinite(se) && abs(val) > 0 ? 100 * se / abs(val) : NaN
            println(io, @sprintf("%s,%.6g,%.6g,%.1f", name, val, se, rse))
        end
        n_eta = size(result.omega, 1)
        n_chol = n_eta * (n_eta + 1) ÷ 2
        k = 1
        for j in 1:n_eta, i in j:n_eta
            val = result.omega[i, j]
            se  = isempty(result.se_omega) ? NaN : result.se_omega[k]
            rse = isfinite(se) && abs(val) > 0 ? 100 * se / abs(val) : NaN
            label = i == j ? "omega_$(i)$(j)" : "omega_$(i)$(j)"
            println(io, @sprintf("%s,%.6g,%.6g,%.1f", label, val, se, rse))
            k += 1
        end
        for (i, val) in enumerate(result.sigma)
            se  = isempty(result.se_sigma) ? NaN : result.se_sigma[i]
            rse = isfinite(se) && abs(val) > 0 ? 100 * se / abs(val) : NaN
            println(io, @sprintf("sigma_%d,%.6g,%.6g,%.1f", i, val, se, rse))
        end
    end
end

# ---------------------------------------------------------------------------
# fit command
# ---------------------------------------------------------------------------

function cmd_fit(args, cfg)
    opts = resolve_options(args, cfg)

    model_path = args["model"]
    data_path  = args["data"]

    isfile(model_path) || error("Model file not found: $model_path")
    isfile(data_path)  || error("Data file not found: $data_path")

    opts.verbose && @info "Parsing model: $model_path"
    model = parse_model_file(model_path)

    opts.verbose && @info "Loading data: $data_path"
    pop = read_data(data_path; covariate_columns = opts.covariate_columns)
    opts.verbose && @info @sprintf("Loaded %d subjects, %d observations",
        length(pop),
        sum(s -> length(s.observations), pop.subjects))

    opts.verbose && @info @sprintf(
        "Fitting with optimizer=%s, interaction=%s, n_starts=%d",
        opts.optimizer, opts.interaction, opts.n_starts)

    result = fit(model, pop;
        optimizer           = opts.optimizer,
        outer_maxiter       = opts.outer_maxiter,
        outer_gtol          = opts.outer_gtol,
        inner_maxiter       = opts.inner_maxiter,
        inner_tol           = opts.inner_tol,
        interaction         = opts.interaction,
        run_covariance_step = opts.run_covariance_step,
        n_starts            = opts.n_starts,
        global_search       = opts.global_search,
        verbose             = opts.verbose,
    )

    print_results(result)

    # Write output files
    mkpath(opts.output_dir)
    params_path = joinpath(opts.output_dir, "$(opts.prefix)_params.csv")
    sdtab_path  = joinpath(opts.output_dir, "$(opts.prefix)_sdtab.csv")

    write_params_csv(result, params_path)
    @info "Parameters written to $params_path"

    tab = sdtab(result, pop)
    # Write sdtab using basic CSV (Tables.jl is a dep, but CSV may not be loaded)
    open(sdtab_path, "w") do io
        cols = propertynames(tab)
        println(io, join(string.(cols), ","))
        for row in eachrow(tab)
            println(io, join([@sprintf("%.6g", row[c]) for c in cols], ","))
        end
    end
    @info "Diagnostic table written to $sdtab_path"

    return result
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

function main()
    parser = build_arg_parser()
    args   = parse_args(ARGS, parser)

    command = lowercase(args["command"])
    cfg     = load_config(args["config"])

    if command == "fit"
        cmd_fit(args, cfg)
    else
        println(stderr, "Unknown command: $command. Supported commands: fit")
        exit(1)
    end
end

main()
