# Diagnostics

JuliaNLME provides two levels of model diagnostics: **post-fit observation-level metrics** (PRED, IPRED, CWRES, IWRES) computed automatically after every `fit()` call, and the **Visual Predictive Check (VPC)**, a simulation-based graphical diagnostic for assessing overall model adequacy.

---

## Post-Fit Diagnostics

After a successful `fit()`, `fit_saem()`, or `fit_its()` call, the `FitResult` contains per-subject results from which standard pharmacometric diagnostics can be extracted.

### Accessing Diagnostics

[`sdtab`](@ref) returns a long-format DataFrame with one row per observation, mirroring the NONMEM SDTAB output:

```julia
result = fit(model, pop; interaction=true)
tab = sdtab(result, pop)
```

| Column | Description |
|--------|-------------|
| `ID` | Subject identifier |
| `TIME` | Observation time |
| `DV` | Observed dependent variable |
| `PRED` | Population prediction (η = 0) |
| `IPRED` | Individual prediction at EBE η̂ |
| `CWRES` | Conditional Weighted Residual |
| `IWRES` | Individual Weighted Residual |
| `ETA1`, `ETA2`, … | Empirical Bayes Estimates |

### Residual Definitions

**PRED** is the model prediction at the population mean (η = 0):
```math
\text{PRED}_j = f(\hat{\theta}, \eta = 0, x_j)
```

**IPRED** is the model prediction at the individual EBEs:
```math
\text{IPRED}_j = f(\hat{\theta}, \hat{\eta}_i, x_j)
```

**IWRES** (Individual Weighted Residuals) measures how well each observation is described by the individual fit:
```math
\text{IWRES}_j = \frac{y_j - \text{IPRED}_j}{\sqrt{V_j(\text{IPRED}_j)}}
```
where ``V_j`` is the residual variance at the individual prediction. Values outside ±3 indicate potential outliers or model misspecification.

**CWRES** (Conditional Weighted Residuals) uses the FOCE linearized model to account for between-subject variability:
```math
\text{CWRES}_j = \frac{y_j - \tilde{f}_j}{\sqrt{\tilde{R}_{jj}}}
```
where ``\tilde{f}_j = f(\hat{\eta}) - H_i \hat{\eta}`` is the population prediction from the linearized model, and ``\tilde{R} = H_i \Omega H_i^\top + R`` is the total marginal variance. CWRES is preferred over IWRES for detecting systematic trends because it accounts for the full uncertainty structure.

### Goodness-of-Fit Plots

A typical set of diagnostic plots:

```julia
using TidierPlots

tab = sdtab(result, pop)

# DV vs PRED
p1 = ggplot(tab, @aes(x = PRED, y = DV)) +
    geom_point(alpha = 0.6) +
    geom_line(DataFrame(x=[0, max_val], y=[0, max_val]),
              @aes(x=x, y=y), color="red", linetype=:dash) +
    labs(x="PRED", y="DV", title="DV vs PRED")

# DV vs IPRED
p2 = ggplot(tab, @aes(x = IPRED, y = DV)) +
    geom_point(alpha = 0.6) +
    geom_line(DataFrame(x=[0, max_val], y=[0, max_val]),
              @aes(x=x, y=y), color="red", linetype=:dash) +
    labs(x="IPRED", y="DV", title="DV vs IPRED")

# CWRES vs TIME
p3 = ggplot(tab, @aes(x = TIME, y = CWRES)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0.0, color="red", linetype=:dash) +
    labs(x="Time", y="CWRES", title="CWRES vs Time")

# CWRES vs PRED
p4 = ggplot(tab, @aes(x = PRED, y = CWRES)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0.0, color="red", linetype=:dash) +
    labs(x="PRED", y="CWRES", title="CWRES vs PRED")
```

!!! tip "Identity lines"
    `geom_abline` is not available in TidierPlots. Draw an identity line with
    `geom_line` and a two-point DataFrame: `DataFrame(x=[lo, hi], y=[lo, hi])`.

### ETA Shrinkage

ETA shrinkage measures how much the EBEs are pulled toward zero relative to the population distribution:

```math
\text{shrinkage}_k = 1 - \frac{\text{SD}(\hat{\eta}_{ik})}{\sqrt{\Omega_{kk}}}
```

High shrinkage (> 30%) means the individual data are sparse: the EBEs carry little information and IPRED-based diagnostics become unreliable. In that situation, CWRES (which uses the population-level model) is more informative than IWRES.

```julia
# Compute ETA shrinkage manually from sdtab output
using Statistics
n_eta = size(result.omega, 1)
for k in 1:n_eta
    eta_col = Symbol("ETA$k")
    sd_eta  = std(tab[!, eta_col])
    omega_sd = sqrt(result.omega[k, k])
    shrinkage = 1 - sd_eta / omega_sd
    @printf "ETA%d shrinkage: %.1f%%\n" k shrinkage*100
end
```

---

## Visual Predictive Check (VPC)

A VPC is a simulation-based graphical diagnostic that assesses whether the model reproduces the observed data distribution. It is more sensitive to structural misspecification than residual-based diagnostics, particularly for detecting incorrect absorption shapes, terminal slopes, or variability structure.

### Concept

For each simulation replicate:
1. Draw individual parameters η̂ᵢ ~ N(0, Ω̂) for each subject.
2. Compute IPRED at each observation time and sample DV ~ N(IPRED, V).

After many replicates, compute the 5th, 50th, and 95th percentiles of the simulated DV in each time bin, then build a confidence interval across replicates for each percentile. The observed data percentiles are overlaid. If the model is adequate, the observed percentile lines should fall within the simulated confidence bands.

### Generating a VPC

```julia
using JuliaNLME, TidierPlots

# 1. Simulate from the fitted model
sim_df = simulate(model, result, obs_df; n_sims = 500)

# 2. Compute VPC statistics
v = vpc(obs_df, sim_df)

# 3. Plot
p = plot_vpc(v; x_lab="Time (h)", y_lab="Concentration (mg/L)")
ggsave("vpc.png", p, width=600, height=400)
```

The `obs_df` passed to both `simulate` and `vpc` should be the original NONMEM-format DataFrame (with dose records and observation records). Both functions filter to observation records internally.

### How to Read the VPC Plot

- **Blue ribbons**: 90% confidence interval around the simulated 5th, 50th, and 95th percentiles. The width of the ribbon reflects simulation uncertainty — wider bands indicate fewer replicates or sparse data in that time bin.
- **Red dashed lines**: observed 5th and 95th percentiles.
- **Red solid line**: observed median.

A well-fitting model shows the observed lines tracking through the center of the corresponding ribbons. Common patterns indicating misspecification:

| Pattern | Likely cause |
|---------|-------------|
| Observed median above/below sim ribbon | Bias in structural model (wrong CL or V) |
| Observed PI lines outside sim ribbons | Variability misspecified (Ω or σ too small/large) |
| Observed lines crossing ribbons | Incorrect absorption or elimination shape |
| Good central trend, poor PI coverage | BSV underestimated |

### Prediction-Corrected VPC

When subjects have different doses or covariate values, the raw VPC mixes observations from different exposure levels into the same bins. Prediction correction normalises each observation by the per-bin median population prediction:

```math
\text{DV}_\text{corr} = \text{DV} \times \frac{\text{median}(\text{PRED}_\text{bin})}{\text{PRED}}
```

This makes all subjects comparable on the same scale. Use it when the dataset has dose heterogeneity or influential covariates:

```julia
# Prediction correction requires PRED in obs_df.
# Use sdtab() to get PRED for the observed data, then merge back.
tab = sdtab(result, pop)
obs_with_pred = leftjoin(obs_df,
    select(tab, :ID => :ID, :TIME => :TIME, :PRED => :PRED);
    on = [:ID, :TIME])

v = vpc(obs_with_pred, sim_df; pred_corr = true)
p = plot_vpc(v; y_lab = "Pred-corrected Concentration")
```

### Stratified VPC

When the dataset contains a discrete covariate (dose group, formulation, indication), a stratified VPC evaluates the model separately for each stratum:

```julia
v = vpc(obs_df, sim_df; stratify = :DOSE_GROUP)
p = plot_vpc(v)   # facets are added automatically
```

Both `obs_df` and `sim_df` must contain the stratification column.

### VPC Options

**`vpc()` options:**

| Keyword | Default | Description |
|---------|---------|-------------|
| `bins` | `:auto` | Binning method: `:auto`, `:equal`, `:quantile`, or explicit `Vector{Float64}` |
| `n_bins` | `0` (auto) | Number of bins; auto selects `min(max(3, n÷40), 15)` |
| `pi` | `(0.05, 0.95)` | Prediction interval percentiles |
| `ci` | `(0.05, 0.95)` | Confidence interval around each PI percentile |
| `stratify` | `nothing` | Column name(s) to stratify by |
| `pred_corr` | `false` | Apply prediction correction |

**`plot_vpc()` options:**

| Keyword | Default | Description |
|---------|---------|-------------|
| `obs_points` | `true` | Overlay raw observed data as scatter points |
| `sim_median` | `false` | Show median lines of each simulated CI band |
| `log_y` | `false` | Log₁₀ y-axis |
| `x_lab` | `"Time"` | x-axis label |
| `y_lab` | `"Concentration"` | y-axis label |
| `pi_fill` | `"#5B9BD5"` | Fill colour for simulated CI ribbons |
| `obs_line_color` | `"#C0392B"` | Colour for observed percentile lines |
| `ribbon_alpha` | `0.25` | Ribbon transparency |

### Number of Simulations

More replicates give narrower CI ribbons and more reliable VPCs. As a guide:

| Dataset size | Recommended `n_sims` |
|---|---|
| < 50 subjects | 500–1000 |
| 50–200 subjects | 200–500 |
| > 200 subjects | 100–200 |

VPC computation scales as O(n_sims × n_subjects) so there is a practical speed limit for large datasets.
