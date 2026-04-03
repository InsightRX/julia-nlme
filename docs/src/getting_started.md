# Getting Started

This guide walks through fitting a 1-compartment oral PK model to a simulated warfarin dataset.

## Step 1: Define a Model File

Create a file `warfarin_oral.jnlme`:

```
model WarfarinOneCmt

  [parameters]
    theta TVCL(0.134, 0.001, 10.0)
    theta TVV(8.1,   0.1,   100.0)
    theta TVKA(1.0,  0.1,  10.0)

    omega ETA_CL ~ 0.07
    omega ETA_V  ~ 0.02
    omega ETA_KA ~ 0.40

    sigma PROP_ERR ~ 0.01

  [individual_parameters]
    CL = TVCL * exp(ETA_CL)
    V  = TVV  * exp(ETA_V)
    KA = TVKA * exp(ETA_KA)

  [structural_model]
    pk one_cpt_oral(cl=CL, v=V, ka=KA)

  [error_model]
    DV ~ proportional(PROP_ERR)

end
```

## Step 2: Prepare Data

JuliaNLME reads NONMEM-format CSV files with columns: `ID`, `TIME`, `AMT`, `DV`, `EVID`, `MDV`, `CMT`, `RATE`. Column names are case-insensitive.

```julia
using JuliaNLME

pop = read_data("data.csv")
println("Loaded $(length(pop)) subjects")
```

You can also pass a `DataFrame` directly:

```julia
using DataFrames
pop = read_data(df)
```

## Step 3: Fit the Model

The simplest approach uses the convenience method that reads both model and data from files:

```julia
result = fit("warfarin_oral.jnlme", "data.csv")
```

For more control, parse the model and construct initial parameters separately:

```julia
model = parse_model_file("warfarin_oral.jnlme")

omega_init = OmegaMatrix([0.09, 0.04, 0.30], [:ETA_CL, :ETA_V, :ETA_KA])
init_params = ModelParameters(
    [0.2, 10.0, 1.5],
    [:TVCL, :TVV, :TVKA],
    omega_init,
    SigmaMatrix([0.02], [:PROP_ERR])
)

result = fit(model, pop, init_params;
             outer_maxiter = 300,
             run_covariance_step = true,
             verbose = true)
```

## Step 4: Inspect Results

```julia
# NONMEM-style summary
print_results(result)

# Parameter table with SEs and RSE%
parameter_table(result)

# Observations table (PRED, IPRED, CWRES, IWRES, ETAs)
tab = sdtab(result, pop)
```

## Key Options

| Keyword | Default | Description |
|---------|---------|-------------|
| `interaction` | `false` | Use FOCE-I (recommended for proportional error models) |
| `outer_maxiter` | `500` | Maximum outer optimizer iterations |
| `inner_maxiter` | `200` | Maximum inner EBE iterations per subject |
| `run_covariance_step` | `true` | Compute standard errors via Hessian |
| `n_starts` | `1` | Multi-start optimization (LHS-based) |
| `optimizer` | `:lbfgs` | Local optimizer (`:lbfgs`, `:bfgs`, or NLopt symbols) |
| `global_search` | `false` | Run gradient-free global pre-search |
| `verbose` | `true` | Print iteration progress |

## Next Steps

- See the [Model File Format](@ref) for the full `.jnlme` specification
- Browse the examples for 1-cpt, 2-cpt, covariate, and ODE models
- Check the [API Reference](@ref) for all exported functions and types
