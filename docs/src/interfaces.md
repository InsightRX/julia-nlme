# Interfaces

JuliaNLME can be used through multiple interfaces depending on your workflow.

## Julia REPL

The most flexible interface. Load the package in an interactive Julia session and call the API functions directly.

### Start a Session

```bash
julia --project=.
```

```julia
using JuliaNLME
```

### Typical Workflow

```julia
# Parse model and load data
model = parse_model_file("examples/warfarin_oral.jnlme")
pop   = read_data("data.csv")

# Fit with default parameters from the model file
result = fit(model, pop; interaction=true, verbose=true)

# Inspect results interactively
print_results(result)
parameter_table(result)

# Diagnostics table
tab = sdtab(result, pop)

# Access individual results
result.ofv          # objective function value
result.theta        # fixed-effect estimates
result.omega        # between-subject variance-covariance matrix
result.subjects[1].eta   # first subject's empirical Bayes estimates

# Importance sampling for parameter uncertainty
is_result = importance_sampling(result, pop; n_samples=1000)

# Simulate from the fitted model
sim_df = simulate(model, pop, result |> r -> ModelParameters(
    r.theta, r.theta_names,
    OmegaMatrix(r.omega), SigmaMatrix(r.sigma)))
```

### Advantages

- Full access to all API functions and keyword arguments
- Inspect intermediate objects (`Population`, `CompiledModel`, `FitResult`)
- Combine with other Julia packages (DataFrames, plotting, etc.)
- Interactive exploration of results and diagnostics

## Command-Line Interface (CLI)

For batch runs, scripted pipelines, or non-interactive workflows. The CLI is a Julia script at `bin/jnlme.jl`.

### Basic Usage

```bash
julia --project=. bin/jnlme.jl fit model.jnlme data.csv
```

This parses the model, loads the data, runs FOCE estimation, prints results to stdout, and writes output files.

### Options

| Flag | Description |
|------|-------------|
| `-O`, `--optimizer NAME` | Optimizer: `lbfgs` (default), `bfgs`, `LD_SLSQP`, `LD_MMA` |
| `-n`, `--iterations N` | Maximum outer iterations (default: 500) |
| `--interaction` | Use FOCE-I (recommended for proportional error models) |
| `--no-covariance` | Skip covariance step (faster, no standard errors) |
| `--n-starts N` | Multi-start optimization with LHS sampling |
| `--global-search` | Enable global pre-search before local optimizer |
| `--covariates COLS` | Comma-separated covariate columns (e.g., `wt,age,sex`) |
| `-o`, `--output-dir DIR` | Output directory (default: current directory) |
| `--prefix STR` | Filename prefix for output files (default: model basename) |
| `-c`, `--config FILE` | Config file path (default: auto-detect `.jnlme` in working dir) |
| `-q`, `--quiet` | Suppress iteration progress |

### Examples

```bash
# Fit with FOCE-I and the SLSQP optimizer
julia --project=. bin/jnlme.jl fit model.jnlme data.csv --interaction -O LD_SLSQP

# Multi-start with 5 runs, output to a results directory
julia --project=. bin/jnlme.jl fit model.jnlme data.csv --n-starts 5 -o results/

# Quiet mode, skip covariance step
julia --project=. bin/jnlme.jl fit model.jnlme data.csv -q --no-covariance
```

### Output Files

The CLI writes two files to the output directory:

- **`<prefix>_params.csv`** — Parameter estimates with standard errors and RSE%
- **`<prefix>_sdtab.csv`** — Diagnostic table (ID, TIME, DV, PRED, IPRED, CWRES, IWRES, ETAs)

### Config File

Place a `.jnlme` TOML file in the working directory to set defaults. CLI arguments always override config values.

```toml
[estimation]
optimizer = "lbfgs"
outer_maxiter = 500
interaction = false
run_covariance_step = true
n_starts = 1

[data]
covariate_columns = ["wt", "age"]

[output]
directory = "results"
prefix = "run01"
```

## R Interface

!!! note "Coming Soon"
    An R-based interface for JuliaNLME is under development. It will allow R users
    to call JuliaNLME estimation routines from R via JuliaCall, combining the
    familiar R pharmacometrics ecosystem with JuliaNLME's estimation engine.

    Stay tuned for updates.
