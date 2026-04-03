# API Reference

## Model Fitting

```@docs
fit
fit_saem
fit_its
```

## Post-Estimation

```@docs
importance_sampling
simulate
```

## Data I/O

```@docs
read_data
```

## Model Parsing

```@docs
parse_model_file
parse_model_string
```

## Output and Diagnostics

```@docs
print_results
parameter_table
sdtab
```

## VPC

```@docs
vpc
plot_vpc
VPCResult
```

## Residual Functions

```@docs
residual_variance
compute_R_diag
iwres
cwres
```

## Parameterization

```@docs
pack_params
unpack_params
```

## PK Equations

```@docs
one_cpt_iv_bolus
one_cpt_infusion
one_cpt_oral
two_cpt_iv_bolus
two_cpt_infusion
two_cpt_oral
three_cpt_iv_bolus
three_cpt_infusion
three_cpt_oral
predict_subject
```

## Types

```@docs
Population
Subject
DoseEvent
ModelParameters
OmegaMatrix
SigmaMatrix
CompiledModel
FitResult
SubjectResult
ISResult
ODESpec
```
