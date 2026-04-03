# JuliaNLME.jl

*Non-Linear Mixed Effects modeling for pharmacokinetics in Julia.*

## Overview

JuliaNLME provides population pharmacokinetic (PK) model estimation using the **FOCE** (First-Order Conditional Estimation) and **SAEM** (Stochastic Approximation Expectation-Maximization) methods. It is designed for fitting nonlinear mixed-effects models to clinical PK data.

## Features

- **FOCE and FOCE-I estimation** with automatic differentiation (ForwardDiff.jl)
- **SAEM estimation** as an alternative algorithm
- **Built-in analytical PK models**: 1- and 2-compartment with IV bolus, infusion, and oral absorption
- **ODE-based models**: define arbitrary differential equations for nonlinear PK (e.g., Michaelis-Menten elimination)
- **Covariate support**: time-constant and time-varying covariates with automatic detection
- **Model file format** (`.jnlme`): declarative model specification with parameters, individual-level equations, structural model, and error model
- **NONMEM-compatible data format**: reads standard NONMEM CSV datasets
- **Post-estimation diagnostics**: PRED, IPRED, CWRES, IWRES, ETA shrinkage, SDTAB output
- **Visual Predictive Check (VPC)**: simulation-based model adequacy diagnostic with prediction correction and stratification
- **Importance sampling** for parameter uncertainty (standard errors and non-parametric confidence intervals)
- **Multi-start optimization** with Latin Hypercube Sampling
- **Covariance step** via Hessian inversion for standard error computation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/insightrx/julia-nlme.git")
```

## Quick Start

```julia
using JuliaNLME

# Fit a model from files
result = fit("model.jnlme", "data.csv")

# Print NONMEM-style results
print_results(result)

# Parameter estimates with standard errors
parameter_table(result)
```

See the [Getting Started](@ref) guide for a complete walkthrough.

## Supported PK Models

| Symbol | Route | Required Parameters |
|--------|-------|-------------------|
| `:one_cpt_iv_bolus` | IV bolus | `cl, v` |
| `:one_cpt_infusion` | IV infusion | `cl, v` |
| `:one_cpt_oral` | Oral | `cl, v, ka` (optional: `f`) |
| `:two_cpt_iv_bolus` | IV bolus | `cl, v1, q, v2` |
| `:two_cpt_infusion` | IV infusion | `cl, v1, q, v2` |
| `:two_cpt_oral` | Oral | `cl, v1, q, v2, ka` (optional: `f`) |
| `:three_cpt_iv_bolus` | IV bolus | `cl, v1, q2, v2, q3, v3` |
| `:three_cpt_infusion` | IV infusion | `cl, v1, q2, v2, q3, v3` |
| `:three_cpt_oral` | Oral | `cl, v1, q2, v2, q3, v3, ka` (optional: `f`) |
| ODE (custom) | Any | User-defined via `[odes]` block |
