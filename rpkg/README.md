# julianlme — R interface to JuliaNLME

R wrapper around [JuliaNLME](https://insightrx.github.io/julia-nlme), providing
`jnlme_fit()` and `jnlme_simulate()` for population PK/PD modeling from R.

## Prerequisites

- **R** ≥ 4.0
- **Julia** ≥ 1.9 (on PATH or location given via `julia_home`)
- **JuliaNLME** installed in a Julia project (see main repo)
- R package **JuliaCall** (`install.packages("JuliaCall")`)

## Installation

```r
# Install JuliaCall if not already installed
install.packages("JuliaCall")

# Install julianlme from source (run from the repo root)
install.packages("rpkg", repos = NULL, type = "source")
```

Or with `devtools`:

```r
devtools::install("rpkg")
```

## Quick Start

```r
library(julianlme)

# 1. Initialize Julia and JuliaNLME (once per session)
jnlme_setup(
  project  = "/path/to/julia-nlme",  # Julia project with JuliaNLME
  nthreads = 4                        # optional: parallel per-subject computation
)

# 2. Parse a model file
model <- jnlme_model("examples/warfarin_oral.jnlme")
print(model)

# 3. Load data (NONMEM format)
data <- read.csv("data/warfarin.csv")

# 4. Fit with FOCE-I
result <- jnlme_fit(model, data, interaction = TRUE)
print(result)

# Access estimates
result$theta     # named numeric vector
result$omega     # named matrix
result$ofv       # objective function value
result$aic       # AIC

# Per-subject diagnostics (id, ipred, pred, iwres, cwres)
head(result$diagnostics)

# Empirical Bayes Estimates
head(result$etas)

# 5. Simulate from the fitted model
sim <- jnlme_simulate(model, data, result)

# 100 replicates for VPC
vpc <- jnlme_simulate(model, data, result, n_sims = 100)
```

## Estimation Methods

```r
# FOCE / FOCE-I (default)
result <- jnlme_fit(model, data, interaction = TRUE)

# ITS (fast warm-start)
result_its <- jnlme_fit_its(model, data)

# SAEM (robust for complex models)
result_saem <- jnlme_fit_saem(model, data)
```

## Threading

Start Julia with multiple threads for faster estimation, especially with ODE
models or large datasets:

```r
jnlme_setup(project = "/path/to/julia-nlme", nthreads = 4)

# Then pass nthreads to the fit call
result <- jnlme_fit(model, data, nthreads = 4)
```

## How It Works

julianlme uses [JuliaCall](https://non-contradiction.github.io/JuliaCall/) to
embed Julia in the R session. Data is transferred via temporary CSV files —
R writes the dataset to a temp file, Julia reads it as a `DataFrame`, runs
estimation, and returns results as a named R list. No RCall.jl dependency is
required.

The Julia-side bridge (`inst/julia/bridge.jl`) caches compiled model and fit
result objects between calls so re-fitting or simulating from the same model
does not re-parse the model file. Call `jnlme_clear_cache()` to release memory
between unrelated analyses.
