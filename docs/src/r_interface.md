# Running from R

The `julianlme` R package wraps JuliaNLME's `fit()` and `simulate()` functions, letting you fit and simulate population PK/PD models from an R session without writing any Julia code. It uses [JuliaCall](https://non-contradiction.github.io/JuliaCall/) to embed a Julia runtime inside R.

## Installation

### Prerequisites

- **Julia** ≥ 1.9, on PATH or accessible via the `julia_home` argument
- **JuliaNLME** installed in a Julia project (see [Getting Started](@ref))
- R package **JuliaCall**: `install.packages("JuliaCall")`

### Install julianlme

The package lives in the `rpkg/` directory of the JuliaNLME repository:

```r
# With devtools
devtools::install("/path/to/julia-nlme/rpkg")

# Or from source
install.packages("/path/to/julia-nlme/rpkg", repos = NULL, type = "source")
```

## First-Time Setup

Call `jnlme_setup()` once at the start of each R session. It starts the Julia
runtime, activates the JuliaNLME project, and loads the bridge functions.

```r
library(julianlme)

jnlme_setup(
  project  = "/path/to/julia-nlme",   # folder containing Project.toml
  nthreads = 1                         # increase for parallel per-subject computation
)
```

### Startup Time

The first time `jnlme_setup()` is called on a machine, Julia precompiles
JuliaNLME and its dependencies to native code. This is a one-time cost that
writes a compiled cache to `~/.julia/compiled/`.

| Event | First ever run | Later sessions |
|---|---|---|
| Julia startup | ~3–5 s | ~3–5 s (always) |
| `using JuliaNLME` | **30–90 s** (one-time precompilation) | ~2–5 s (cache) |
| First `jnlme_fit()` call | ~10–30 s (JIT compilation) | ~1–3 s (cache) |

### Eliminating Startup Overhead with a Sysimage

A **sysimage** is a precompiled native snapshot of Julia + JuliaNLME. Loading
it reduces total startup to ~1–2 s every session. Build one with
`jnlme_build_sysimage()` — a one-time operation:

```r
# Takes 5–15 minutes; run once after installing or updating JuliaNLME
sysimage <- jnlme_build_sysimage(
  path    = "julianlme.so",          # use .dll on Windows
  project = "/path/to/julia-nlme"
)
```

Then use the sysimage in every future session:

```r
jnlme_setup(
  project       = "/path/to/julia-nlme",
  sysimage_path = "julianlme.so"
)
```

`jnlme_build_sysimage()` installs PackageCompiler.jl if needed, then runs a
warmup script that exercises all key code paths (FOCE, FOCE-I, ITS, SAEM, and
`simulate()`). All JIT-compiled specializations — including ForwardDiff dual
arithmetic — are baked into the image.

## Parsing a Model

`jnlme_model()` parses a `.jnlme` file and returns an R object holding the
compiled model and its default parameter values. The same model file format is
used as in the Julia API.

```r
model <- jnlme_model("/path/to/julia-nlme/examples/warfarin_oral.jnlme")
print(model)
```

```
JuliaNLME model: WarfarinOneCmt
  PK model:     one_cpt_oral
  Error model:  proportional
  THETA:        TVCL, TVV, TVKA
  ETA:          ETA_CL, ETA_V, ETA_KA
  SIGMA:        PROP_ERR
```

The default initial parameter values from the `[parameters]` block are
available as named R vectors and matrices:

```r
model$theta_init   # named numeric vector
model$theta_lower  # lower bounds
model$theta_upper  # upper bounds
model$omega_init   # named matrix (BSV covariance)
model$sigma_init   # named numeric vector
```

## Fitting a Model

All three estimation methods are supported. Each takes a `jnlme_model` and a
NONMEM-format `data.frame` and returns a `jnlme_fit` object.

### FOCE / FOCE-I

```r
result <- jnlme_fit(
  model,
  data,
  interaction         = TRUE,   # FOCE-I: recommended for proportional/combined error
  run_covariance_step = TRUE,
  verbose             = TRUE
)
```

### ITS (fast warm-start)

```r
result_its <- jnlme_fit_its(model, data)
```

### SAEM (robust for complex models)

```r
result_saem <- jnlme_fit_saem(model, data)
```

## Inspecting Results

All fit functions return a `jnlme_fit` S3 object. Printing it gives a
NONMEM-style summary:

```r
print(result)
```

```
JuliaNLME fit [converged]
  Model:      WarfarinOneCmt
  Method:     FOCE-I
  OFV: -142.3    AIC: -128.3    BIC: -110.5
  Subjects: 10   Observations: 110   Parameters: 7   Iterations: 47

THETA:
       Estimate     SE   RSE%
TVCL     0.1338 0.0084    6.3
TVV      8.0521 0.3412    4.2
TVKA     1.0183 0.1024   10.1

OMEGA (variance-covariance):
        ETA_CL  ETA_V ETA_KA
ETA_CL  0.0682  0.019  0.000
ETA_V   0.0190  0.021  0.000
ETA_KA  0.0000  0.000  0.381

SIGMA:
          Estimate     SE   RSE%
PROP_ERR    0.0098 0.0014   14.7
```

Individual fields:

```r
result$theta        # named numeric vector of fixed-effect estimates
result$omega        # named matrix (BSV covariance)
result$sigma        # named numeric vector (residual variance)
result$se_theta     # standard errors for theta (NULL if covariance step failed)
result$ofv          # objective function value (−2 log-likelihood)
result$aic
result$bic
result$converged    # logical
result$n_subjects
result$n_obs
result$n_iterations
result$warnings     # character vector of any warnings issued during fitting
```

S3 generics:

```r
coef(result)        # same as result$theta
vcov(result)        # covariance matrix of packed parameters
```

### Per-Subject Diagnostics

`result$diagnostics` is a long-format data.frame with one row per observation:

```r
head(result$diagnostics)
#   id    ipred     pred      iwres      cwres
# 1  1 0.512453 0.478324  0.234123  0.182453
# 2  1 0.982345 0.912453 -0.123457 -0.083453
```

`result$etas` is a wide-format data.frame with one row per subject:

```r
result$etas
#   id  eta_ETA_CL   eta_ETA_V  eta_ETA_KA
# 1  1  0.08212412  0.04123456 -0.24123456
# 2  2 -0.12345679 -0.07890123  0.12345678
```

Merge these with your original data.frame for plotting:

```r
obs        <- data[data$EVID == 0, ]
obs$IPRED  <- result$diagnostics$ipred
obs$PRED   <- result$diagnostics$pred
obs$CWRES  <- result$diagnostics$cwres

library(ggplot2)

# DV vs IPRED
ggplot(obs, aes(x = IPRED, y = DV)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  theme_bw()

# CWRES vs TIME
ggplot(obs, aes(x = TIME, y = CWRES)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_hline(yintercept = c(-2, 2), color = "grey50", linetype = "dotted") +
  theme_bw()
```

## Simulating

`jnlme_simulate()` draws individual parameters from the population
distribution and simulates observations. It takes the fitted model and result
and returns a data.frame.

```r
# Single simulation replicate
sim <- jnlme_simulate(model, data, result)

# 200 replicates for a visual predictive check (VPC)
vpc <- jnlme_simulate(model, data, result, n_sims = 200)
```

The returned data.frame has one row per observation per replicate, with
columns `id`, `time`, `dv` (simulated), `pred`, `ipred`, and `_sim`
(replicate index).

```r
library(dplyr)

# 5th/50th/95th prediction interval across replicates
pi <- vpc |>
  group_by(time) |>
  summarise(p05 = quantile(dv, 0.05),
            p50 = quantile(dv, 0.50),
            p95 = quantile(dv, 0.95))

ggplot() +
  geom_ribbon(data = pi, aes(x = time, ymin = p05, ymax = p95),
              fill = "steelblue", alpha = 0.3) +
  geom_line(data = pi, aes(x = time, y = p50), color = "steelblue") +
  geom_point(data = data[data$EVID == 0, ], aes(x = TIME, y = DV)) +
  theme_bw()
```

## Multi-Threading

Pass `nthreads` to both `jnlme_setup()` and the fit functions. The Julia
runtime must be started with multiple threads — `jnlme_setup(nthreads = N)`
sets `JULIA_NUM_THREADS` before starting Julia.

```r
jnlme_setup(project = "/path/to/julia-nlme", nthreads = 4)

result <- jnlme_fit(model, data, nthreads = 4)
```

See the [Multi-Threading](estimation_methods.md#Multi-Threading) section of
Estimation Methods for guidance on expected speedups by dataset size and model
type.

## Memory Management

Model and fit result objects are cached on the Julia side by a UUID key held
in the `jnlme_model` and `jnlme_fit` R objects. The cache persists for the
lifetime of the Julia session. Call `jnlme_clear_cache()` to release memory
between unrelated analyses:

```r
jnlme_clear_cache()
```

## Function Reference

| Function | Description |
|---|---|
| `jnlme_setup()` | Initialize Julia and load JuliaNLME |
| `jnlme_build_sysimage()` | Build a precompiled sysimage for fast startup |
| `jnlme_model()` | Parse a `.jnlme` model file |
| `jnlme_fit()` | Fit with FOCE or FOCE-I |
| `jnlme_fit_its()` | Fit with ITS |
| `jnlme_fit_saem()` | Fit with SAEM |
| `jnlme_simulate()` | Simulate observations from a fitted model |
| `jnlme_clear_cache()` | Release Julia-side object cache |
