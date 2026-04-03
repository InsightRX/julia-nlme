# Example 1: Warfarin 1-Compartment Oral

This example fits a 1-compartment oral PK model to a simulated warfarin dataset with 10 subjects.

**Source**: [`examples/ex1_warfarin.jl`](https://github.com/insightrx/julia-nlme/blob/main/examples/ex1_warfarin.jl)

## Model

The `.jnlme` model file ([`warfarin_oral.jnlme`](https://github.com/insightrx/julia-nlme/blob/main/examples/warfarin_oral.jnlme)):

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

## Simulate Data

```julia
using JuliaNLME, DataFrames, Random

Random.seed!(42)

true_theta = [0.134, 8.1, 1.0]   # TVCL, TVV, TVKA
true_omega = [0.07, 0.02, 0.40]  # BSV variances
true_sigma = [0.01]               # proportional error variance

obs_times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0]

function simulate_subject(id, dose, times, theta, omega_var, sigma_var)
    cl = theta[1] * exp(sqrt(omega_var[1]) * randn())
    v  = theta[2] * exp(sqrt(omega_var[2]) * randn())
    ka = theta[3] * exp(sqrt(omega_var[3]) * randn())
    rows = []
    push!(rows, (ID=id, TIME=0.0, AMT=dose, DV=missing, EVID=1, MDV=1, CMT=1, RATE=0.0))
    for t in times
        ipred = one_cpt_oral(; cl=cl, v=v, ka=ka, dose=dose, t=t)
        dv = ipred * (1 + sqrt(sigma_var[1]) * randn())
        push!(rows, (ID=id, TIME=t, AMT=missing, DV=max(dv, 0.01), EVID=0, MDV=0, CMT=1, RATE=0.0))
    end
    return rows
end

all_rows = []
for id in 1:10
    append!(all_rows, simulate_subject(id, 100.0, obs_times, true_theta, true_omega, true_sigma))
end
df = DataFrame(all_rows)
```

## Fit the Model

```julia
pop = read_data(df)

omega_init = OmegaMatrix([0.09, 0.04, 0.30], [:ETA_CL, :ETA_V, :ETA_KA])
init_params = ModelParameters(
    [0.2, 10.0, 1.5],
    [:TVCL, :TVV, :TVKA],
    omega_init,
    SigmaMatrix([0.02], [:PROP_ERR])
)

model = parse_model_file("examples/warfarin_oral.jnlme")

result = fit(model, pop, init_params;
             outer_maxiter = 300,
             run_covariance_step = true,
             optimizer = :LD_SLSQP,
             verbose = true)
```

## Inspect Results

```julia
print_results(result)
parameter_table(result)

tab = sdtab(result, pop)
```

## Run It

```bash
julia --project=. examples/ex1_warfarin.jl
```
