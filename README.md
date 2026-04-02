# JuliaNLME

Non-Linear Mixed Effects (NLME) estimation in Julia using the FOCE/FOCE-I method. Designed for population pharmacokinetic (PK) modeling with a NONMEM-compatible data format and a simple model DSL.

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/insightrx/julia-nlme")
```

Requires Julia 1.9+.

## Quick start

### Command-line interface

```bash
# Fit a model to a CSV dataset
bin/jnlme fit model.jnlme data.csv

# With options
bin/jnlme fit model.jnlme data.csv --interaction --optimizer LD_SLSQP -o results/

# See all options
bin/jnlme --help
```

Output files are written to the current directory (or `--output-dir`):
- `{model}_params.csv` — parameter estimates, standard errors, %RSE
- `{model}_sdtab.csv` — per-observation diagnostics (DV, PRED, IPRED, CWRES, ETAs)

**Fast startup.** Julia's first run compiles everything (~10 s). To reduce this to ~1 s, build a precompiled system image once:
```bash
julia --project=. scripts/build_sysimage.jl   # ~10 min, one-time
bin/jnlme fit ...                              # ~1 s afterwards
```

### Julia API

```julia
using JuliaNLME

model  = parse_model_file("model.jnlme")
pop    = read_data("data.csv")
result = fit(model, pop; interaction=true, verbose=true)

print_results(result)
display(parameter_table(result))
tab = sdtab(result, pop)   # DataFrame: DV, PRED, IPRED, CWRES, ETAs
```

## Model file format (`.jnlme`)

```
model ModelName

  [parameters]
    theta TVCL(0.134, 0.001, 10.0)   # name(initial, lower, upper)
    theta TVV(8.1, 0.1, 100.0)
    theta TVKA(1.0, 0.1, 10.0)

    omega ETA_CL ~ 0.07              # scalar BSV (variance)
    omega ETA_V  ~ 0.02
    omega ETA_KA ~ 0.40

    sigma PROP_ERR ~ 0.01            # residual error variance

  [individual_parameters]
    CL = TVCL * exp(ETA_CL)          # plain Julia math
    V  = TVV  * exp(ETA_V)
    KA = TVKA * exp(ETA_KA)

  [structural_model]
    pk one_cpt_oral(cl=CL, v=V, ka=KA)

  [error_model]
    DV ~ proportional(PROP_ERR)

end
```

### Parameter syntax

| Syntax | Description |
|---|---|
| `theta NAME(init, lower, upper)` | Fixed-effect parameter with bounds |
| `theta NAME(init)` | Fixed-effect, bounds default to (1e-9, Inf) |
| `theta NAME(init, lower, upper) fix` | Hold constant during estimation |
| `omega ETA_NAME ~ value` | Scalar BSV variance |
| `omega [ETA_A, ETA_B] ~ [var_a, cov_ab, var_b]` | Full covariance block |
| `sigma NAME ~ value` | Residual variance |
| `sigma NAME ~ value fix` | Fix residual variance |

Uppercase identifiers in `[individual_parameters]` that are not theta/eta names are treated as covariates (e.g. `WT`, `CRCL`). They are looked up in the dataset by their lowercase name.

### Supported PK models

| Symbol | Required parameters |
|---|---|
| `one_cpt_iv_bolus` | `cl, v` |
| `one_cpt_infusion` | `cl, v` |
| `one_cpt_oral` | `cl, v, ka` |
| `two_cpt_iv_bolus` | `cl, v1, q, v2` |
| `two_cpt_infusion` | `cl, v1, q, v2` |
| `two_cpt_oral` | `cl, v1, q, v2, ka` |

### Error models

| Symbol | Description |
|---|---|
| `proportional(SIGMA)` | `DV = IPRED * (1 + ε)` |
| `additive(SIGMA)` | `DV = IPRED + ε` |
| `combined(SIGMA_PROP, SIGMA_ADD)` | Proportional + additive |

## Covariates with model file example

```
model TwoCptCov
  [parameters]
    theta TVCL(5.0, 0.1, 100.0)
    theta TVV1(50.0, 1.0, 500.0)
    theta TVQ(10.0, 0.1, 200.0)
    theta TVV2(100.0, 1.0, 1000.0)
    theta TVKA(1.2, 0.01, 10.0)
    theta THETA_WT(0.75, 0.1, 2.0)
    theta THETA_CRCL(0.50, 0.0, 2.0)

    omega ETA_CL ~ 0.10
    omega ETA_V1 ~ 0.10

    sigma PROP_ERR ~ 0.02

  [individual_parameters]
    CL = TVCL * (WT / 70.0)^THETA_WT * (CRCL / 100.0)^THETA_CRCL * exp(ETA_CL)
    V1 = TVV1 * (WT / 70.0)^THETA_WT * exp(ETA_V1)
    Q  = TVQ
    V2 = TVV2
    KA = TVKA

  [structural_model]
    pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

  [error_model]
    DV ~ proportional(PROP_ERR)

end
```

Time-varying covariates (e.g. creatinine clearance measured at each visit) are automatically detected from the dataset and interpolated via last-observation-carried-forward (LOCF) at each observation time.

## Data format

NONMEM-style CSV with columns (case-insensitive):

| Column | Description |
|---|---|
| `ID` | Subject identifier |
| `TIME` | Time since first dose |
| `AMT` | Dose amount (for dosing records) |
| `DV` | Observed concentration (for observation records) |
| `EVID` | Event type: 1 = dose, 0 = observation |
| `MDV` | Missing DV flag: 1 = ignore, 0 = use |
| `CMT` | Compartment (optional, defaults to 1) |
| `RATE` | Infusion rate (optional, 0 = bolus) |
| `*` | Any additional columns are treated as covariates |

Loading:
```julia
# Auto-detect covariates
pop = read_data("data.csv")

# Specify covariate columns explicitly (avoids picking up string columns)
pop = read_data("data.csv"; covariate_columns = [:wt, :age, :crcl])
```

## Estimation options

```julia
result = fit(model, pop;
    optimizer           = :lbfgs,    # :lbfgs | :bfgs | :LD_SLSQP | :LD_MMA | ...
    interaction         = false,     # true = FOCE-I (recommended for proportional/combined)
    outer_maxiter       = 500,
    run_covariance_step = true,      # compute standard errors
    n_starts            = 1,         # >1 = Latin Hypercube multi-start
    global_search       = false,     # GN_CRS2_LM global pre-search
    verbose             = true,
)
```

**Optimizer choice.** `:lbfgs` (default) works well for most problems. For complex models or when results are sensitive to starting values, try `:LD_SLSQP` (NLopt) or increase `n_starts`.

## Config file for the CLI

Place a `.jnlme` file in your working directory to set project-level defaults:

```toml
[estimation]
optimizer    = "lbfgs"
interaction  = true
outer_maxiter = 500
n_starts     = 1

[output]
directory = "results"
```

CLI arguments always override the config file. See `examples/.jnlme` for a fully annotated template.

## Running examples

```bash
julia --project=. examples/ex1_warfarin.jl        # 1-CMT oral, simulated data
julia --project=. examples/ex2_two_cpt_iv.jl      # 2-CMT IV infusion
julia --project=. examples/ex3_two_cpt_oral_cov.jl # 2-CMT oral with WT and CRCL covariates
```

## Running tests

```bash
julia --project=. test/runtests.jl
```
