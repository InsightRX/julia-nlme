# Example 4: ODE-Based Michaelis-Menten Model

This example demonstrates ODE-based estimation for a model with saturable (Michaelis-Menten) elimination that cannot be expressed with analytical solutions.

**Source**: [`examples/ex4_ode_mm.jl`](https://github.com/insightrx/julia-nlme/blob/main/examples/ex4_ode_mm.jl)

## Background

At high concentrations (C >> KM), elimination approaches a fixed rate VMAX rather than being proportional to concentration. This gives a convex terminal slope on a log-concentration plot, which is distinctly nonlinear compared to first-order models.

## Model File

The `.jnlme` file ([`mm_oral.jnlme`](https://github.com/insightrx/julia-nlme/blob/main/examples/mm_oral.jnlme)):

```
model MMOral

  [parameters]
    theta TVVMAX(3.0, 0.1, 50.0)
    theta TVKM(5.0, 0.1, 100.0)
    theta TVV(10.0, 1.0, 200.0)
    theta TVKA(1.2, 0.05, 20.0)

    omega ETA_VMAX ~ 0.15
    omega ETA_V    ~ 0.10

    sigma PROP_ERR ~ 0.02

  [individual_parameters]
    VMAX = TVVMAX * exp(ETA_VMAX)
    KM   = TVKM
    V    = TVV   * exp(ETA_V)
    KA   = TVKA

  [structural_model]
    ode(obs_cmt=central, states=[depot, central])

  [odes]
    d/dt(depot)   = -KA * depot
    d/dt(central) = KA * depot / V - VMAX * central / (KM + central)

  [error_model]
    DV ~ proportional(PROP_ERR)

end
```

## Key Differences from Analytical Models

1. **`[structural_model]`** uses `ode(...)` instead of `pk model_name(...)`
2. **`[odes]`** block defines the differential equations
3. **`obs_cmt=central`** specifies which state is the observable (mapped to DV)
4. All variables from `[individual_parameters]` are available in the ODE expressions (accessed as `p.NAME`)

## Fitting

```julia
using JuliaNLME, DataFrames

pop = read_data(df)
model = parse_model_file("examples/mm_oral.jnlme")

result = fit(model, pop;
             interaction = true,
             outer_maxiter = 400,
             run_covariance_step = true,
             verbose = true)

print_results(result)
parameter_table(result)
```

The model uses the default initial parameters from the `[parameters]` block when no explicit `init_params` is provided.

## ODE Solver Details

- **Solver**: `Tsit5()` (5th-order explicit Runge-Kutta) via OrdinaryDiffEq.jl
- **Dose handling**: bolus doses are applied as instantaneous state increments, with the ODE integrated segment-by-segment between dose events
- **ForwardDiff compatibility**: state types are promoted from the parameter types, so Dual numbers propagate through the ODE solver automatically

## Run It

```bash
julia --project=. examples/ex4_ode_mm.jl
```
