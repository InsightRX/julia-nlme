# Example 2: Two-Compartment IV Bolus

This example fits a 2-compartment IV bolus PK model to 15 simulated subjects and demonstrates diagonal vs. full omega estimation.

**Source**: [`examples/ex2_two_cpt_iv.jl`](https://github.com/insightrx/julia-nlme/blob/main/examples/ex2_two_cpt_iv.jl)

## Model

Uses [`two_cpt_iv.jnlme`](https://github.com/insightrx/julia-nlme/blob/main/examples/two_cpt_iv.jnlme) with 4 parameters (CL, V1, Q, V2) and log-normal between-subject variability on all parameters.

## Setup and Fit

```julia
using JuliaNLME, DataFrames, Random

Random.seed!(123)

# True parameters
true_theta = [5.0, 15.0, 3.0, 30.0]   # TVCL, TVV1, TVQ, TVV2
true_omega = [0.10, 0.10, 0.10, 0.10]  # BSV variances
true_sigma = [0.01]                     # proportional error

# ... simulate 15 subjects with IV bolus 100 mg ...

pop = read_data(df)
model = parse_model_file("examples/two_cpt_iv.jnlme")

omega_init = OmegaMatrix([0.15, 0.15, 0.15, 0.15],
                          [:ETA_CL, :ETA_V1, :ETA_Q, :ETA_V2])
init_params = ModelParameters(
    [4.0, 12.0, 2.0, 25.0],
    [:TVCL, :TVV1, :TVQ, :TVV2],
    omega_init,
    SigmaMatrix([0.02], [:PROP_ERR])
)

result = fit(model, pop, init_params;
             outer_maxiter = 500,
             run_covariance_step = true,
             optimizer = :LD_SLSQP,
             verbose = true)

print_results(result)
```

## Diagonal vs. Full Omega

By default, `OmegaMatrix` estimates the full lower-triangular Cholesky factor (all covariances). Pass `diagonal=true` to restrict to independent variances only:

```julia
omega_diag = OmegaMatrix([0.15, 0.15, 0.15, 0.15],
                          [:ETA_CL, :ETA_V1, :ETA_Q, :ETA_V2];
                          diagonal = true)
init_diag = ModelParameters(
    [4.0, 12.0, 2.0, 25.0],
    [:TVCL, :TVV1, :TVQ, :TVV2],
    omega_diag,
    SigmaMatrix([0.02], [:PROP_ERR])
)

result_diag = fit(model, pop, init_diag;
                  outer_maxiter = 500,
                  run_covariance_step = true,
                  verbose = false)
```

Compare models using AIC:

```julia
using Printf
@printf "Full OMEGA  (10 params): OFV = %.3f  AIC = %.3f\n" result.ofv result.aic
@printf "Diag OMEGA  ( 4 params): OFV = %.3f  AIC = %.3f\n" result_diag.ofv result_diag.aic
```

## Run It

```bash
julia --project=. examples/ex2_two_cpt_iv.jl
```
