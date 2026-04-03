# Example 3: Two-Compartment Oral with Covariates

This example demonstrates covariate modeling with both time-constant (weight) and time-varying (creatinine clearance) covariates in a 2-compartment oral PK model.

**Source**: [`examples/ex3_two_cpt_oral_cov.jl`](https://github.com/insightrx/julia-nlme/blob/main/examples/ex3_two_cpt_oral_cov.jl)

## Covariate Model

```
CL = TVCL * (WT / 70)^THETA_WT * (CRCL / 100)^THETA_CRCL * exp(ETA_CL)
V1 = TVV1 * (WT / 70)^THETA_WT * exp(ETA_V1)
```

- **WT** (weight): time-constant covariate with allometric scaling
- **CRCL** (creatinine clearance): time-varying covariate that declines over the study period, simulating acute kidney injury

## Model File

The `.jnlme` file ([`two_cpt_oral_cov.jnlme`](https://github.com/insightrx/julia-nlme/blob/main/examples/two_cpt_oral_cov.jnlme)):

```
model TwoCptOralCov

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
    omega ETA_Q  ~ 0.05
    omega ETA_V2 ~ 0.05
    omega ETA_KA ~ 0.15

    sigma PROP_ERR ~ 0.02

  [individual_parameters]
    CL = TVCL * (WT / 70.0)^THETA_WT * (CRCL / 100.0)^THETA_CRCL * exp(ETA_CL)
    V1 = TVV1 * (WT / 70.0)^THETA_WT * exp(ETA_V1)
    Q  = TVQ  * exp(ETA_Q)
    V2 = TVV2 * exp(ETA_V2)
    KA = TVKA * exp(ETA_KA)

  [structural_model]
    pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

  [error_model]
    DV ~ proportional(PROP_ERR)

end
```

## Covariate Handling

`read_data` automatically detects whether a covariate is time-constant or time-varying:

```julia
pop = read_data(df)
s1 = pop[1]
println("Time-constant covariates: $(keys(s1.covariates))")  # (:wt,)
println("Time-varying covariates:  $(keys(s1.tvcov))")       # (:crcl,)
```

In the model file, covariates are written in UPPERCASE (`WT`, `CRCL`). The parser maps these to the lowercase keys stored in the dataset.

## Fitting with FOCE-I

Proportional error models benefit from the interaction term:

```julia
result = fit(model, pop, init_params;
             interaction = true,
             optimizer = :bfgs,
             verbose = true)

print_results(result)
parameter_table(result)
```

## Run It

```bash
julia --project=. examples/ex3_two_cpt_oral_cov.jl
```
