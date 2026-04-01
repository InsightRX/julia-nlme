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
end

struct OmegaSpec
    names::Vector{Symbol}
    values::Vector{Float64}   # diagonal if length == 1, or lower-triangle
end

struct SigmaSpec
    name::Symbol
    value::Float64
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

    for line in lines
        line = strip(line)
        startswith(line, '#') && continue

        # theta NAME(init, lower, upper) or theta NAME(init)
        m = match(r"^theta\s+(\w+)\(([^)]+)\)", line)
        if m !== nothing
            name = Symbol(m.captures[1])
            parts = split(m.captures[2], ',')
            init  = parse(Float64, strip(parts[1]))
            lower = length(parts) >= 2 ? parse(Float64, strip(parts[2])) : 1e-9
            upper = length(parts) >= 3 ? parse(Float64, strip(parts[3])) : Inf
            push!(thetas, ThetaSpec(name, init, lower, upper))
            continue
        end

        # omega [ETA_A, ETA_B, ...] ~ [val1, val2, val3]  (block)
        m = match(r"^omega\s+\[([^\]]+)\]\s*~\s*\[([^\]]+)\]", line)
        if m !== nothing
            names  = [Symbol(strip(s)) for s in split(m.captures[1], ',')]
            values = [parse(Float64, strip(s)) for s in split(m.captures[2], ',')]
            push!(omegas, OmegaSpec(names, values))
            continue
        end

        # omega ETA_NAME ~ value  (scalar)
        m = match(r"^omega\s+(\w+)\s*~\s*([\d.eE+\-]+)", line)
        if m !== nothing
            name  = Symbol(m.captures[1])
            value = parse(Float64, m.captures[2])
            push!(omegas, OmegaSpec([name], [value]))
            continue
        end

        # sigma NAME ~ value
        m = match(r"^sigma\s+(\w+)\s*~\s*([\d.eE+\-]+)", line)
        if m !== nothing
            name  = Symbol(m.captures[1])
            value = parse(Float64, m.captures[2])
            push!(sigmas, SigmaSpec(name, value))
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
    theta_lower = [s.lower   for s in theta_specs]
    theta_upper = [s.upper   for s in theta_specs]

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

    omega = OmegaMatrix(omega_mat, eta_names)
    sigma = SigmaMatrix([s.value for s in sigma_specs],
                         [s.name  for s in sigma_specs])

    return ModelParameters(theta_vals, theta_names, theta_lower, theta_upper, omega, sigma)
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
    pk_model, pk_param_map                = _parse_structural_model(blocks["structural_model"])
    error_model, sigma_names              = _parse_error_model(blocks["error_model"])

    fn_body, eta_names, _ = _codegen_pk_param_fn(
        blocks["individual_parameters"],
        theta_specs, omega_specs, pk_param_map
    )

    # Build a full lambda Expr and compile via RuntimeGeneratedFunctions
    # to avoid world-age issues from eval().
    fn_expr = Expr(:->,
                   Expr(:tuple, :theta, :eta, :covariates),
                   fn_body)
    pk_param_fn = @RuntimeGeneratedFunction(fn_expr)

    theta_names   = [s.name for s in theta_specs]
    n_epsilon     = length(sigma_specs)
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
        default_params
    )
end
