"""
Lightweight parser for `.jnlme` model files.

Rather than a full grammar, we parse the block structure and use
`Meta.parse` on the `[individual_parameters]` block (plain Julia math).

Block layout:
  model <Name>
    [description]   (optional, ignored)
    [parameters]
    [individual_parameters]
    [structural_model]
    [error_model]
  end
"""

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------

"""
Extract named blocks from model file content.
Returns Dict{String, Vector{String}} mapping block name → lines.
"""
function _extract_blocks(content::AbstractString)
    blocks = Dict{String, Vector{String}}()
    current_block = nothing
    model_name = "UnnamedModel"

    for line in split(content, '\n')
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, '#') && continue

        # model <Name>
        m = match(r"^model\s+(\w+)", stripped)
        if m !== nothing
            model_name = m.captures[1]
            continue
        end

        # [block_name]
        m = match(r"^\[(\w+)\]", stripped)
        if m !== nothing
            current_block = m.captures[1]
            blocks[current_block] = String[]
            continue
        end

        if stripped == "end"
            current_block = nothing
            continue
        end

        if current_block !== nothing
            push!(blocks[current_block], stripped)
        end
    end

    return model_name, blocks
end

# ---------------------------------------------------------------------------
# [parameters] block parser
# ---------------------------------------------------------------------------

struct ThetaSpec
    name::Symbol
    initial::Float64
    lower::Float64
    upper::Float64
    fixed::Bool
end

struct OmegaSpec
    names::Vector{Symbol}
    values::Vector{Float64}   # diagonal if length == 1, or lower-triangle
    fixed::Bool
end

struct SigmaSpec
    name::Symbol
    value::Float64
    fixed::Bool
end

"""
Parse [parameters] block lines.

Supported syntax:
  theta NAME(init, lower, upper)
  theta NAME(init)
  omega ETA_NAME ~ value
  omega [ETA_A, ETA_B] ~ [var_a, cov_ab, var_b]
  sigma NAME ~ value
"""
function _parse_parameters(lines::Vector{String})
    thetas = ThetaSpec[]
    omegas = OmegaSpec[]
    sigmas = SigmaSpec[]

    _is_fixed(line) = occursin(r"\bfix(ed)?\b"i, line)

    for line in lines
        line = strip(line)
        startswith(line, '#') && continue
        fixed = _is_fixed(line)

        # theta NAME(init, lower, upper) or theta NAME(init)
        m = match(r"^theta\s+(\w+)\(([^)]+)\)", line)
        if m !== nothing
            name = Symbol(m.captures[1])
            parts = split(m.captures[2], ',')
            init  = parse(Float64, strip(parts[1]))
            lower = length(parts) >= 2 ? parse(Float64, strip(parts[2])) : 1e-9
            upper = length(parts) >= 3 ? parse(Float64, strip(parts[3])) : Inf
            push!(thetas, ThetaSpec(name, init, lower, upper, fixed))
            continue
        end

        # omega [ETA_A, ETA_B, ...] ~ [val1, val2, val3]  (block)
        m = match(r"^omega\s+\[([^\]]+)\]\s*~\s*\[([^\]]+)\]", line)
        if m !== nothing
            names  = [Symbol(strip(s)) for s in split(m.captures[1], ',')]
            values = [parse(Float64, strip(s)) for s in split(m.captures[2], ',')]
            push!(omegas, OmegaSpec(names, values, fixed))
            continue
        end

        # omega ETA_NAME ~ value  (scalar)
        m = match(r"^omega\s+(\w+)\s*~\s*([\d.eE+\-]+)", line)
        if m !== nothing
            name  = Symbol(m.captures[1])
            value = parse(Float64, m.captures[2])
            push!(omegas, OmegaSpec([name], [value], fixed))
            continue
        end

        # sigma NAME ~ value
        m = match(r"^sigma\s+(\w+)\s*~\s*([\d.eE+\-]+)", line)
        if m !== nothing
            name  = Symbol(m.captures[1])
            value = parse(Float64, m.captures[2])
            push!(sigmas, SigmaSpec(name, value, fixed))
            continue
        end
    end

    return thetas, omegas, sigmas
end

# ---------------------------------------------------------------------------
# [structural_model] block parser
# ---------------------------------------------------------------------------

"""
Parse `pk <model_symbol>(key=VAR, ...)` declaration.
Returns (pk_model_symbol, Dict{Symbol=>Symbol} param_map).
"""
function _parse_structural_model(lines::Vector{String})
    for line in lines
        m = match(r"^\s*pk\s+(\w+)\s*\(([^)]+)\)", line)
        if m !== nothing
            model_sym = Symbol(m.captures[1])
            param_map = Dict{Symbol, Symbol}()
            for pair in split(m.captures[2], ',')
                kv = split(pair, '=')
                length(kv) == 2 || continue
                param_map[Symbol(strip(kv[1]))] = Symbol(strip(kv[2]))
            end
            return model_sym, param_map
        end
    end
    error("No `pk` declaration found in [structural_model] block")
end

# ---------------------------------------------------------------------------
# [error_model] block parser
# ---------------------------------------------------------------------------

"""
Parse `DV ~ proportional(SIGMA_NAME)` etc.
Returns (error_model_symbol, sigma_names).
"""
function _parse_error_model(lines::Vector{String})
    for line in lines
        m = match(r"^\w+\s*~\s*(\w+)\(([^)]+)\)", line)
        if m !== nothing
            model_name  = lowercase(m.captures[1])
            sigma_names = [Symbol(strip(s)) for s in split(m.captures[2], ',')]
            model_sym   = Symbol(model_name)
            model_sym in (:additive, :proportional, :combined) ||
                error("Unknown error model: $model_name")
            return model_sym, sigma_names
        end
    end
    error("No error model declaration found in [error_model] block")
end

# ---------------------------------------------------------------------------
# [individual_parameters] code generation
# ---------------------------------------------------------------------------

"""
Generate a Julia function from the individual_parameters block.

The block is plain Julia expressions like:
  CL = TVCL * exp(ETA_CL) * (WT/70)^0.75

We wrap it in a function:
  (theta, eta, covariates) -> begin ... return (cl=CL, v=V, ...) end

Where theta/eta are accessed by position via named aliases, and
covariates are Dict{Symbol, T}.
"""
function _codegen_pk_param_fn(lines::Vector{String},
                                theta_specs::Vector{ThetaSpec},
                                omega_specs::Vector{OmegaSpec},
                                pk_param_map::Dict{Symbol, Symbol})

    # Build theta name → index map
    theta_idx = Dict(spec.name => i for (i, spec) in enumerate(theta_specs))

    # Build eta name → index map (from omega specs, in order)
    eta_names_ordered = Symbol[]
    for spec in omega_specs, name in spec.names
        push!(eta_names_ordered, name)
    end
    eta_idx = Dict(name => i for (i, name) in enumerate(eta_names_ordered))

    # Preamble: assign theta names
    preamble = Expr[]
    for (name, idx) in theta_idx
        push!(preamble, :($name = theta[$idx]))
    end

    # Assign eta names
    for (name, idx) in eta_idx
        push!(preamble, :($name = eta[$idx]))
    end

    # Assign covariates (accessed from Dict)
    # We'll inject them dynamically — collect referenced covariate names
    # by finding identifiers not in theta/eta
    known = Set(vcat(keys(theta_idx)..., keys(eta_idx)...))

    # Parse body lines
    body_exprs = Expr[]
    defined = Set{Symbol}()
    for line in lines
        line = strip(line)
        isempty(line) || startswith(line, '#') && continue
        expr = Meta.parse(line)
        expr isa Expr || continue
        push!(body_exprs, expr)
        # Track defined variable names
        if expr.head == :(=)
            push!(defined, expr.args[1])
        end
    end

    # Return named tuple with PK parameters (from pk_param_map)
    # AST for (cl=CL, v=V) uses head :tuple with Expr(:(=), key, val) args
    ret_args = [Expr(:(=), pk_key, pk_var_name)
                for (pk_key, pk_var_name) in pk_param_map]
    ret_expr = Expr(:tuple, ret_args...)

    # Covariate access: inject `COV = covariates[:COV]` for any identifier
    # that appears in body but is not theta/eta/Julia-builtin
    cov_names = Symbol[]
    for line in lines
        # Rough heuristic: extract all identifiers
        for m in eachmatch(r"\b([A-Z][A-Z0-9_]+)\b", line)
            s = Symbol(m.captures[1])
            s in known && continue
            s in defined && continue
            push!(cov_names, s)
        end
    end
    cov_names = unique(cov_names)

    cov_preamble = [:($(name) = covariates[$(QuoteNode(Symbol(lowercase(string(name)))))]) for name in cov_names]

    # Assemble full function body
    all_exprs = vcat(preamble, cov_preamble, body_exprs, [ret_expr])
    fn_body   = Expr(:block, all_exprs...)

    return fn_body, eta_names_ordered, cov_names
end

# ---------------------------------------------------------------------------
# Build ModelParameters from parsed specs
# ---------------------------------------------------------------------------

function _build_init_params(theta_specs::Vector{ThetaSpec},
                              omega_specs::Vector{OmegaSpec},
                              sigma_specs::Vector{SigmaSpec})
    theta_vals  = [s.initial for s in theta_specs]
    theta_names = [s.name    for s in theta_specs]
    # Fixed theta: collapse lower == upper == init so the optimizer cannot move it
    theta_lower = [s.fixed ? s.initial : s.lower for s in theta_specs]
    theta_upper = [s.fixed ? s.initial : s.upper for s in theta_specs]

    eta_names = Symbol[]
    for spec in omega_specs, name in spec.names
        push!(eta_names, name)
    end
    n_eta = length(eta_names)

    omega_mat = zeros(n_eta, n_eta)
    for spec in omega_specs
        n = length(spec.names)
        if n == 1
            i = findfirst(==(spec.names[1]), eta_names)
            omega_mat[i, i] = spec.values[1]
        else
            idxs = [findfirst(==(name), eta_names) for name in spec.names]
            k = 1
            for col in 1:n, row in col:n
                i, j = idxs[row], idxs[col]
                omega_mat[i, j] = spec.values[k]
                omega_mat[j, i] = spec.values[k]
                k += 1
            end
        end
    end

    # Use diagonal packing when every omega spec is a scalar (no covariance blocks).
    # This ensures pack_params and packed_fixed agree on the number of Cholesky elements.
    # Mixed or block specs (e.g. omega [ETA_A, ETA_B] ~ [...]) require full packing.
    all_scalar = all(spec -> length(spec.names) == 1, omega_specs)
    omega = OmegaMatrix(omega_mat, eta_names; diagonal = all_scalar)
    sigma = SigmaMatrix([s.value for s in sigma_specs],
                         [s.name  for s in sigma_specs])

    # Build packed_fixed: marks which elements of the packed vector are frozen.
    # Layout mirrors pack_params: [theta..., chol(Ω)..., log(σ)...]
    n_theta = length(theta_specs)
    n_chol  = omega.diagonal ? n_eta : n_eta * (n_eta + 1) ÷ 2
    n_sigma = length(sigma_specs)
    packed_fixed = falses(n_theta + n_chol + n_sigma)

    # Theta section
    for (i, spec) in enumerate(theta_specs)
        packed_fixed[i] = spec.fixed
    end

    # Omega Cholesky section — mark all elements of a fixed spec's block
    chol_idx = n_theta + 1
    for spec in omega_specs
        n = length(spec.names)
        n_elems = omega.diagonal ? 1 : n * (n + 1) ÷ 2
        if spec.fixed
            packed_fixed[chol_idx : chol_idx + n_elems - 1] .= true
        end
        chol_idx += n_elems
    end

    # Sigma section
    for (i, spec) in enumerate(sigma_specs)
        packed_fixed[n_theta + n_chol + i] = spec.fixed
    end

    return ModelParameters(theta_vals, theta_names, theta_lower, theta_upper,
                            omega, sigma, packed_fixed)
end

# ---------------------------------------------------------------------------
# ODE structural model helpers
# ---------------------------------------------------------------------------

"""
Parse `ode(obs_cmt=X, states=[A, B, ...])` from [structural_model].
Returns `(state_names, obs_cmt_name)`.
"""
function _parse_ode_structural(lines::Vector{String})
    for line in lines
        # Match the full ode(...) with possibly nested brackets
        m = match(r"^\s*ode\s*\((.+)\)\s*$", line)
        m === nothing && continue

        args_str = m.captures[1]

        # Extract obs_cmt=<name>
        m_obs = match(r"\bobs_cmt\s*=\s*(\w+)", args_str)
        m_obs !== nothing || error("ode() missing obs_cmt= argument")
        obs_cmt = Symbol(m_obs.captures[1])

        # Extract states=[A, B, ...]
        m_states = match(r"\bstates\s*=\s*\[([^\]]+)\]", args_str)
        m_states !== nothing || error("ode() missing states=[...] argument")
        state_names = [Symbol(strip(s)) for s in split(m_states.captures[1], ',')]

        obs_cmt in state_names || error("obs_cmt=$obs_cmt not found in states=$state_names")
        return state_names, obs_cmt
    end
    error("No `ode(...)` declaration found in [structural_model] block")
end

"""
Collect all variable names assigned in an [individual_parameters] block.
"""
function _collect_defined_vars(lines::Vector{String})
    defined = Symbol[]
    for line in lines
        line = strip(line)
        isempty(line) || startswith(line, '#') && continue
        expr = Meta.parse(line)
        if expr isa Expr && expr.head == :(=)
            push!(defined, expr.args[1])
        end
    end
    return defined
end

"""
Parse the [odes] block and return an ODESpec.

Each line must have the form `d/dt(state_name) = <expression>`.
The generated function is out-of-place with signature `(u, p, t) → SVector(...)`,
compiled via RuntimeGeneratedFunctions to avoid world-age issues.

Using an out-of-place SVector function allows OrdinaryDiffEq to use its
static-array fast path, which eliminates heap allocations during the ODE
integration. This is critical for performance when ForwardDiff Dual numbers
flow through the ODE (as occurs during FOCE gradient computation).
"""
function _parse_odes_block(lines::Vector{String},
                            state_names::Vector{Symbol},
                            obs_cmt_name::Symbol,
                            param_names::Vector{Symbol})

    state_idx   = Dict(name => i for (i, name) in enumerate(state_names))
    obs_cmt_idx = state_idx[obs_cmt_name]

    # Parse each d/dt(X) = rhs line
    equations = Dict{Symbol, Expr}()
    for line in lines
        line = strip(line)
        isempty(line) || startswith(line, '#') && continue
        # Strip trailing comment
        line = strip(split(line, '#')[1])
        m = match(r"^d/dt\((\w+)\)\s*=\s*(.+)$", line)
        m !== nothing || error("Cannot parse ODE equation: \"$line\"")
        state = Symbol(m.captures[1])
        state in state_names || error("Unknown state variable: $state")
        equations[state] = Meta.parse(strip(m.captures[2]))
    end

    # Build out-of-place function body:
    #   (u, p, t) -> begin
    #       depot = u[1]; central = u[2]   # state aliases
    #       KA = p.KA; VMAX = p.VMAX; ...  # param aliases
    #       _ode_d_depot   = -KA * depot
    #       _ode_d_central = KA * depot / V - VMAX * central / (KM + central)
    #       return SVector(_ode_d_depot, _ode_d_central)
    #   end
    body = Expr[]

    # State aliases: depot = u[1], central = u[2], ...
    for (name, idx) in state_idx
        push!(body, :($name = u[$idx]))
    end

    # Parameter aliases: KA = p.KA, VMAX = p.VMAX, ...
    for name in param_names
        push!(body, :($name = p.$name))
    end

    # Derivative variables (avoid clobbering state/param names)
    deriv_vars = [Symbol("_ode_d_", state_names[i]) for i in 1:length(state_names)]

    # ODE equations: _ode_d_depot = rhs, etc.
    for (state, rhs) in equations
        idx = state_idx[state]
        push!(body, :($(deriv_vars[idx]) = $rhs))
    end

    # Default 0 for any state without an explicit equation
    for (name, idx) in state_idx
        if !haskey(equations, name)
            push!(body, :($(deriv_vars[idx]) = zero(eltype(u))))
        end
    end

    # Return SVector of derivatives — enables OrdinaryDiffEq static-array path
    push!(body, :(return SVector($(deriv_vars...))))

    fn_body = Expr(:block, body...)
    fn_expr = Expr(:->, Expr(:tuple, :u, :p, :t), fn_body)   # out-of-place: 3 args
    ode_fn  = @RuntimeGeneratedFunction(fn_expr)

    return ODESpec(ode_fn, state_names, obs_cmt_idx)
end

# ---------------------------------------------------------------------------
# Top-level parse function
# ---------------------------------------------------------------------------

"""
    parse_model_file(path)

Parse a `.jnlme` model file and return a `CompiledModel`.
"""
function parse_model_file(path::AbstractString)::CompiledModel
    content = read(path, String)
    return parse_model_string(content)
end

"""
    parse_model_string(content)

Parse model DSL from a string. Useful for testing.
"""
function parse_model_string(content::AbstractString)::CompiledModel
    model_name, blocks = _extract_blocks(content)

    haskey(blocks, "parameters")             || error("Missing [parameters] block")
    haskey(blocks, "individual_parameters")  || error("Missing [individual_parameters] block")
    haskey(blocks, "structural_model")       || error("Missing [structural_model] block")
    haskey(blocks, "error_model")            || error("Missing [error_model] block")

    theta_specs, omega_specs, sigma_specs = _parse_parameters(blocks["parameters"])
    error_model, sigma_names              = _parse_error_model(blocks["error_model"])

    # Determine whether this is an ODE or analytical model
    is_ode = any(l -> occursin(r"^\s*ode\s*\(", l), blocks["structural_model"])

    if is_ode
        haskey(blocks, "odes") || error("ODE model requires an [odes] block")

        state_names, obs_cmt_name = _parse_ode_structural(blocks["structural_model"])

        # pk_param_fn returns ALL defined individual params for ODE models
        defined_vars = _collect_defined_vars(blocks["individual_parameters"])
        pk_param_map = Dict{Symbol, Symbol}(v => v for v in defined_vars)

        fn_body, eta_names, _ = _codegen_pk_param_fn(
            blocks["individual_parameters"], theta_specs, omega_specs, pk_param_map)

        fn_expr = Expr(:->, Expr(:tuple, :theta, :eta, :covariates), fn_body)
        pk_param_fn = @RuntimeGeneratedFunction(fn_expr)

        param_names = collect(keys(pk_param_map))
        ode_spec    = _parse_odes_block(blocks["odes"], state_names, obs_cmt_name, param_names)
        pk_model    = :ode

    else
        pk_model, pk_param_map = _parse_structural_model(blocks["structural_model"])

        fn_body, eta_names, _ = _codegen_pk_param_fn(
            blocks["individual_parameters"], theta_specs, omega_specs, pk_param_map)

        fn_expr = Expr(:->, Expr(:tuple, :theta, :eta, :covariates), fn_body)
        pk_param_fn = @RuntimeGeneratedFunction(fn_expr)

        ode_spec = nothing
    end

    theta_names    = [s.name for s in theta_specs]
    n_epsilon      = length(sigma_specs)
    default_params = _build_init_params(theta_specs, omega_specs, sigma_specs)

    return CompiledModel(
        model_name,
        pk_model,
        error_model,
        pk_param_fn,
        length(theta_specs),
        length(eta_names),
        n_epsilon,
        theta_names,
        eta_names,
        default_params,
        ode_spec
    )
end
