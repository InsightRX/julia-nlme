# Model File Format

JuliaNLME uses `.jnlme` model files — a declarative format for specifying population PK models. Models are parsed with [`parse_model_file`](@ref) or [`parse_model_string`](@ref).

## General Structure

```
model ModelName

  [parameters]
    # Fixed effects, random effects, residual error

  [individual_parameters]
    # Equations mapping population params + ETAs to individual PK params

  [structural_model]
    # PK model specification (analytical or ODE)

  [odes]
    # (ODE models only) Differential equations

  [error_model]
    # Residual error model

end
```

## `[parameters]` Block

### Fixed Effects (theta)

```
theta NAME(initial_value, lower_bound, upper_bound)
```

Example:
```
theta TVCL(5.0, 0.1, 100.0)
theta TVV(50.0, 1.0, 500.0)
```

### Random Effects (omega)

Each omega line defines one ETA with its initial variance:

```
omega ETA_NAME ~ variance
```

Example:
```
omega ETA_CL ~ 0.10
omega ETA_V  ~ 0.05
```

When all omegas are specified as individual scalar lines, the resulting omega matrix is diagonal.

### Residual Error (sigma)

```
sigma NAME ~ variance
```

Example:
```
sigma PROP_ERR ~ 0.01
sigma ADD_ERR  ~ 1.0
```

## `[individual_parameters]` Block

Plain Julia expressions mapping population parameters (thetas), random effects (ETAs), and covariates to individual PK parameters. These are compiled into efficient functions at parse time.

```
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
```

### Covariates

Covariates are referenced in UPPERCASE in the model file (e.g., `WT`, `CRCL`). The parser automatically detects these and maps them to the lowercase keys stored in the dataset:

```
[individual_parameters]
  CL = TVCL * (WT / 70.0)^THETA_WT * (CRCL / 100.0)^THETA_CRCL * exp(ETA_CL)
  V1 = TVV1 * (WT / 70.0)^THETA_WT * exp(ETA_V1)
```

Both time-constant and time-varying covariates are supported. `read_data` automatically detects time-varying covariates (those whose values change within a subject).

## `[structural_model]` Block

### Analytical Models

Reference a built-in PK model with parameter mapping:

```
[structural_model]
  pk model_symbol(param1=VAR1, param2=VAR2, ...)
```

Available models:

| Symbol | Parameters |
|--------|-----------|
| `one_cpt_iv_bolus` | `cl, v` |
| `one_cpt_infusion` | `cl, v` |
| `one_cpt_oral` | `cl, v, ka` (optional: `f`) |
| `two_cpt_iv_bolus` | `cl, v1, q, v2` |
| `two_cpt_infusion` | `cl, v1, q, v2` |
| `two_cpt_oral` | `cl, v1, q, v2, ka` (optional: `f`) |
| `three_cpt_iv_bolus` | `cl, v1, q2, v2, q3, v3` |
| `three_cpt_infusion` | `cl, v1, q2, v2, q3, v3` |
| `three_cpt_oral` | `cl, v1, q2, v2, q3, v3, ka` (optional: `f`) |

Example:
```
[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)
```

### ODE Models

For models that cannot be expressed analytically:

```
[structural_model]
  ode(obs_cmt=STATE_NAME, states=[STATE1, STATE2, ...])
```

- `obs_cmt`: the state variable used as the observable (mapped to DV)
- `states`: list of all ODE state variables

Example:
```
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
```

## `[odes]` Block

Required only for ODE models. Define one differential equation per state:

```
[odes]
  d/dt(STATE) = expression
```

The expression can reference any variable defined in `[individual_parameters]`, plus the state variables themselves.

Example (Michaelis-Menten elimination):
```
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - VMAX * central / (KM + central)
```

Doses with `EVID=1` are applied as bolus additions to the first state in the `states` list (or the state matching the `CMT` column).

## `[error_model]` Block

Specifies the residual error model for the dependent variable:

```
DV ~ proportional(SIGMA_NAME)
DV ~ additive(SIGMA_NAME)
DV ~ combined(SIGMA_PROP, SIGMA_ADD)
```

- **Proportional**: `Var = (IPRED * sigma)^2`
- **Additive**: `Var = sigma`
- **Combined**: `Var = (IPRED * sigma_prop)^2 + sigma_add`

## Complete Example

### Analytical Model

```
model WarfarinOneCmt

  [parameters]
    theta TVCL(0.134, 0.001, 10.0)
    theta TVV(8.1, 0.1, 100.0)
    theta TVKA(1.0, 0.1, 10.0)

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

### ODE Model

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
