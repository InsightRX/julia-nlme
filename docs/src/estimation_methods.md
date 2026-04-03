# Estimation Methods

JuliaNLME provides three population parameter estimation algorithms -- FOCE/FOCE-I, SAEM, and ITS -- plus importance sampling for post-estimation parameter uncertainty. All methods produce a [`FitResult`](@ref) with the same structure, and OFV/AIC/BIC values are directly comparable across methods.

## FOCE and FOCE-I

The First-Order Conditional Estimation (FOCE) method is the primary estimation algorithm in JuliaNLME, accessed via [`fit`](@ref). It is the same method used by NONMEM and is the standard approach for population PK modeling.

### How It Works

FOCE is a nested two-loop optimization:

**Outer loop** (population parameters): BFGS or L-BFGS optimization over the packed parameter vector `(theta, chol(omega), log(sigma))`.

**Inner loop** (per-subject, parallelized): for each subject, find the empirical Bayes estimates (EBEs) by minimizing the individual negative log-likelihood:

```math
\text{NLL}_i(\eta) = \frac{1}{2} \left[ \eta^\top \Omega^{-1} \eta + \log|\Omega| + \sum_j \left( \frac{(y_j - f_j)^2}{V_j} + \log V_j \right) \right]
```

where ``f_j`` are the model predictions at the individual parameters ``\theta \cdot \exp(\eta)`` and ``V_j`` is the residual variance. The inner optimization uses ForwardDiff for gradients and the BFGS algorithm.

After finding ``\hat{\eta}_i``, the Jacobian ``H_i = \partial f / \partial \eta |_{\hat{\eta}_i}`` is computed. This is used in the FOCE approximation of the marginal likelihood.

### FOCE vs FOCE-I

The key difference is where the residual variance ``R`` is evaluated:

**FOCE** (`interaction=false`, default): linearizes around the population prediction. The residual variance is evaluated at the linearized prediction ``f_0 = f(\hat{\eta}) - H \hat{\eta}``:

```math
\text{NLL}_i = \frac{1}{2} \left[ (y - f_0)^\top \tilde{R}^{-1} (y - f_0) + \log|\tilde{R}| \right]
```

where ``\tilde{R} = H \Omega H^\top + R(f_0)``.

**FOCE-I** (`interaction=true`): evaluates the residual variance at the individual prediction (IPRED), capturing the eta-epsilon interaction:

```math
\text{NLL}_i = \frac{1}{2} \left[ (y - \text{IPRED})^\top V^{-1} (y - \text{IPRED}) + \hat{\eta}^\top \Omega^{-1} \hat{\eta} + \log|\tilde{R}| \right]
```

where ``\tilde{R} = H \Omega H^\top + V`` and ``V = R(\text{IPRED})``.

!!! tip "When to use FOCE-I"
    Use `interaction=true` when your error model is **proportional** or **combined**. For these error models, the residual variance depends on the predicted concentration, so the interaction between random effects (eta) and residual error (epsilon) matters. For **additive** error models, FOCE and FOCE-I are equivalent.

### Gradient Computation

The outer gradient uses a key property of FOCE: the EBEs are held fixed when differentiating the population objective with respect to ``(\theta, \Omega, \sigma)``. ForwardDiff differentiates through the likelihood computation (not through the inner optimization), which is the standard FOCE approach. The combined `fdfg!` function evaluates the objective and gradient from the same inner-loop result, ensuring consistency for BFGS convergence.

### Covariance Step

After convergence, standard errors are computed via the inverse of the Hessian of the OFV at the optimum (the "sandwich estimator"). This uses `ForwardDiff.hessian` applied to the packed population objective.

### Usage

```julia
# Standard FOCE
result = fit(model, pop, init_params; interaction=false)

# FOCE-I (recommended for proportional/combined error)
result = fit(model, pop, init_params; interaction=true)

# With optimizer options
result = fit(model, pop, init_params;
    interaction = true,
    optimizer = :lbfgs,        # :lbfgs, :bfgs, or NLopt symbols
    outer_maxiter = 500,
    outer_gtol = 1e-6,
    inner_maxiter = 200,
    inner_tol = 1e-8,
    run_covariance_step = true,
    verbose = true)
```

### Multi-Start Optimization

For models with complex likelihood surfaces, multi-start optimization can help avoid local minima:

```julia
result = fit(model, pop, init_params;
    n_starts = 10,           # 10 starting points via Latin Hypercube Sampling
    global_search = true)    # optional: gradient-free global pre-search
```

The first starting point is always the user-provided initial parameters. The remaining `n_starts - 1` points are generated via Latin Hypercube Sampling across the parameter bounds in packed (log-scale) space, guaranteeing uniform marginal coverage. The result with the lowest OFV is polished with a final single-start fit to obtain the covariance step.

### Available Optimizers

| Symbol | Algorithm | Notes |
|--------|-----------|-------|
| `:lbfgs` | Limited-memory BFGS | Default. Memory-efficient, good for many parameters |
| `:bfgs` | Full BFGS | Stores full Hessian approximation |
| `:LD_SLSQP` | NLopt Sequential Least Squares | Handles box constraints natively |
| `:LD_MMA` | NLopt Method of Moving Asymptotes | Alternative gradient-based |
| `:LD_TNEWTON_PRECOND_RESTART` | NLopt Truncated Newton | For large-scale problems |

## SAEM

The Stochastic Approximation Expectation-Maximization (SAEM) algorithm is an alternative estimation method accessed via [`fit_saem`](@ref). It can be more robust than FOCE for models with complex random-effects structures or when FOCE has convergence difficulties.

### How It Works

SAEM uses a two-phase iterative scheme:

**Phase 1 — Exploration** (`n_iter_exploration` iterations, default 150): the step size ``\gamma_k = 1``, allowing rapid exploration of the parameter space.

**Phase 2 — Convergence** (`n_iter_convergence` iterations, default 250): the step size ``\gamma_k = 1/(k - K_1)`` decreases, ensuring almost-sure convergence to the maximum likelihood estimate.

Each iteration consists of:

1. **Simulation step**: Metropolis-Hastings (MH) sampling of ``\eta_i`` from the conditional posterior ``p(\eta_i | y_i; \theta, \Omega, \sigma)`` for each subject. The proposal is ``\eta_\text{prop} = \eta_\text{curr} + \delta_i \cdot L_\Omega \cdot z`` where ``z \sim N(0, I)`` and ``\delta_i`` is an adaptive per-subject step size targeting a 40% acceptance rate.

2. **Stochastic approximation update** of the sufficient statistic for ``\Omega``:
```math
s_2 \leftarrow (1 - \gamma) s_2 + \gamma \cdot \frac{1}{N} \sum_i \eta_i \eta_i^\top
```

3. **M-step for** ``\Omega`` (closed form): ``\Omega_k = s_2``

4. **M-step for** ``(\theta, \sigma)`` via BFGS optimization of the conditional observation NLL with ETAs held fixed at their sampled values.

### Final OFV

After SAEM convergence, the final OFV is computed using the FOCE/Laplace approximation (with a full inner-loop EBE optimization). This ensures that AIC and BIC values are directly comparable between `fit()` and `fit_saem()` results.

### When to Use SAEM

- Models where FOCE has convergence difficulties
- Complex random-effects structures
- As a complementary method to cross-validate FOCE results
- Models with many random effects where the likelihood surface has many local optima

### Usage

```julia
# SAEM with default settings
result = fit_saem(model, pop, init_params)

# With custom iteration counts
result = fit_saem(model, pop, init_params;
    n_iter_exploration = 200,
    n_iter_convergence = 300,
    n_mh_steps = 3,
    run_covariance_step = true,
    interaction = true,
    verbose = true)

# Using model defaults for initial parameters
result = fit_saem(model, pop)
```

### SAEM-Specific Options

| Keyword | Default | Description |
|---------|---------|-------------|
| `n_iter_exploration` | `150` | Phase-1 iterations (``\gamma = 1``) |
| `n_iter_convergence` | `250` | Phase-2 iterations (``\gamma = 1/k``) |
| `n_mh_steps` | `2` | MH sampling steps per subject per iteration |
| `adapt_interval` | `50` | Iterations between MH step-size adaptations |
| `saem_theta_maxiter` | `30` | BFGS iterations for the ``(\theta, \sigma)`` M-step |

## ITS

The Iterative Two-Stage (ITS) algorithm is a fast, deterministic estimation method accessed via [`fit_its`](@ref). It is useful as a quick first estimate or as a warm-start for FOCE, particularly for models where FOCE is slow to converge from default initial parameters.

### How It Works

ITS is a deterministic EM algorithm. Each iteration consists of:

1. **E-step** (per-subject, parallelized): find the MAP estimate ``\hat{\eta}_i`` by minimizing the individual NLL (identical to the FOCE inner loop). Compute the Jacobian ``H_i = \partial f / \partial \eta |_{\hat{\eta}_i}`` and the approximate posterior covariance:

```math
\hat{C}_i = \left[ H_i^\top V_i^{-1} H_i + \Omega^{-1} \right]^{-1}
```

where ``V_i = \text{diag}(R_i)`` is the diagonal residual variance matrix.

2. **M-step for** ``\Omega`` (closed form): using the bias-corrected sample covariance of the EBEs:

```math
\Omega_\text{new} = \frac{1}{N} \sum_i \left[ (\hat{\eta}_i - \bar{\eta})(\hat{\eta}_i - \bar{\eta})^\top + \hat{C}_i \right]
```

The ``\hat{C}_i`` term corrects for EBE shrinkage. Without it, the sample covariance of ``\hat{\eta}_i`` underestimates ``\Omega`` for sparse data.

3. **M-step for** ``(\theta, \sigma)`` via BFGS on the conditional observation NLL with ETAs held fixed.

Convergence is assessed via a rolling window: when the maximum relative change of the running parameter averages over the last `conv_window` iterations falls below `rel_tol`, the algorithm stops.

### Final OFV

After convergence, the final OFV is computed using the FOCE/Laplace approximation (with a full inner-loop EBE optimization), so AIC and BIC values are directly comparable with `fit()` and `fit_saem()`.

### Usage

```julia
# ITS with default settings
result = fit_its(model, pop)

# ITS as a warm-start for FOCE-I
its_result = fit_its(model, pop; run_covariance_step = false, verbose = true)

its_params = ModelParameters(
    its_result.theta,
    model.default_params.theta_names,
    model.default_params.theta_lower,
    model.default_params.theta_upper,
    OmegaMatrix(its_result.omega, model.default_params.omega.eta_names;
                diagonal = model.default_params.omega.diagonal),
    SigmaMatrix(its_result.sigma, model.default_params.sigma.names),
    model.default_params.packed_fixed
)

foce_result = fit(model, pop, its_params; interaction = true)
```

### ITS-Specific Options

| Keyword | Default | Description |
|---------|---------|-------------|
| `n_iter` | `100` | Maximum number of ITS iterations |
| `conv_window` | `20` | Rolling window length for convergence check |
| `rel_tol` | `1e-4` | Relative parameter change tolerance |
| `theta_maxiter` | `30` | BFGS iterations for the ``(\theta, \sigma)`` M-step |

## Importance Sampling

Importance sampling (IS) is a post-estimation method for assessing parameter uncertainty. It is accessed via [`importance_sampling`](@ref) and can be applied to results from either `fit()` or `fit_saem()`.

### Motivation

The standard errors from the covariance step (inverse Hessian) assume that the likelihood surface is well-approximated by a quadratic near the optimum. When this assumption is poor -- for example with small datasets, near-boundary estimates, or skewed parameter distributions -- importance sampling provides more reliable uncertainty estimates including non-parametric confidence intervals.

### How It Works

IS draws ``M`` parameter vectors from a multivariate normal proposal centered at the FOCE estimates:

```math
x^{(m)} = \hat{x} + L_C z^{(m)}, \quad z^{(m)} \sim N(0, I)
```

where ``L_C`` is the Cholesky factor of the FOCE covariance matrix ``\hat{C}``.

For each sample, the full FOCE population likelihood is evaluated (including the inner-loop EBE optimization for all subjects), and IS weights are computed:

```math
\log w^{(m)} = -\text{NLL}_\text{FOCE}(x^{(m)}) + \frac{1}{2} \|z^{(m)}\|^2
```

The normalized weights ``\tilde{w}^{(m)}`` define a weighted distribution over population parameters.

### What IS Produces

The [`ISResult`](@ref) contains:

- **IS OFV**: ``\text{OFV}_\text{IS} = -2(\log \sum w - \log M)``. This marginalizes the FOCE likelihood over parameter uncertainty. A positive ``\Delta\text{OFV}`` (FOCE - IS > 0) indicates the Laplace approximation was conservative.
- **IS-based standard errors**: weighted standard deviation of the parameter samples.
- **IS-weighted medians**: weighted 50th percentile, which can differ from the point estimate when the distribution is skewed.
- **Non-parametric 95% CI**: 2.5th and 97.5th weighted percentiles -- these do not assume symmetry or normality.
- **Effective sample size (ESS)**: measures how well the proposal distribution matches the target. Low ESS (< 20%) suggests the proposal is a poor fit and more samples may be needed.

### Usage

```julia
# Run FOCE first (covariance step required)
result = fit(model, pop; interaction=true, run_covariance_step=true)

# Importance sampling
is_result = importance_sampling(result, pop;
    n_samples = 1000,
    verbose = true)

# Print the full uncertainty table
println(is_result)

# Access individual components
is_result.se_theta       # IS-based SEs for theta
is_result.ci_theta       # 95% CI tuples [(lo, hi), ...]
is_result.median_theta   # IS-weighted medians
is_result.ess_pct        # Effective sample size (%)
is_result.delta_ofv      # OFV_FOCE - OFV_IS
```

### Interpreting IS Results

| Metric | Good | Concern |
|--------|------|---------|
| ESS % | > 30% | < 10% suggests poor proposal fit |
| ``\Delta``OFV | Small (< 5) | Large values suggest asymmetric likelihood |
| SE comparison | IS SE close to FOCE SE | Large differences flag non-quadratic likelihood |
| CI symmetry | Symmetric around estimate | Asymmetric CI indicates skewed distribution |

### Practical Recommendations

- Start with `n_samples=500`, increase to 1000-2000 if ESS is low
- IS is computationally expensive: each sample requires a full inner-loop optimization for all subjects
- Always ensure `run_covariance_step=true` when calling `fit()` or `fit_saem()`, as IS requires the parameter covariance matrix
- Compare IS SEs with FOCE SEs: if they agree, the quadratic approximation is adequate; if they disagree, report the IS estimates

## Multi-Threading

All three estimation methods (`fit`, `fit_saem`, `fit_its`) parallelize per-subject computations across threads. The bottleneck in every method is independent per-subject work: EBE inner-loop optimization, FOCE likelihood summation, MH sampling, or ITS Ĉᵢ computation. These have no inter-subject dependencies and scale well with thread count.

### Enabling Threading

Start Julia with multiple threads and pass `nthreads`:

```julia
# In terminal:  julia -t 4 --project=.

result = fit(model, pop, init_params; nthreads=4)
result = fit_its(model, pop; nthreads=4)
result = fit_saem(model, pop; nthreads=4)
```

The default is `nthreads = Threads.nthreads()` — all threads available to Julia. Pass `nthreads=1` to run sequentially (useful for debugging or reproducibility).

### Guidance by Dataset Size

| Dataset size | Analytical models (1/2/3-cpt) | ODE models |
|---|---|---|
| N < 20 | Modest gain (~1.3–1.8x with 4 threads). Thread overhead is non-trivial relative to per-subject work. | Strong gain (often >2x). ODE integration per subject is expensive regardless of N. |
| N = 20–50 | Good gain (~1.8–2.5x with 4 threads). | Near-linear scaling up to available threads. |
| N > 50 | Strong gain (~2.5–3x with 4 threads). The parallel fraction dominates. | Near-linear scaling; limited mainly by the serial BFGS step. |

!!! note "Amdahl's law"
    Not all work is parallelized. The outer BFGS step, ForwardDiff gradient through the likelihood, and SAEM stochastic approximation update are all sequential. With N=30 analytical subjects the serial fraction is roughly 40%; with N=100 it drops to ~20%, giving better scaling.

### Guidance by Model Complexity

**Analytical 1/2/3-compartment models**: per-subject prediction is closed-form and very fast (microseconds). The inner EBE loop is still the bottleneck, but it converges in few iterations. Threading is worthwhile for N≥20, but the marginal gain per additional thread diminishes past 4.

**ODE models**: each subject requires numerical ODE integration (Tsit5 solver) at every EBE iteration. Per-subject work is orders of magnitude larger than for analytical models, so threading is highly effective even with small N. For ODE models, always use all available threads.

### Typical Performance (4 threads, analytical model)

| N subjects | Speedup vs 1 thread |
|---|---|
| 10 | ~1.5x |
| 30 | ~1.8x |
| 100 | ~2.8x |

ODE models typically achieve 3–4x speedup at the same N values.

### Important Caveats

- **Julia startup with threads**: threads must be set at Julia startup (`julia -t N`). `Threads.nthreads()` cannot be changed after startup.
- **ForwardDiff thread safety**: ForwardDiff v0.10+ is thread-safe. Dual numbers are immutable value types and per-subject computations share no mutable state.
- **BLAS threads**: if your model uses matrix operations inside the structural model, BLAS may use its own thread pool. You can control this with `using LinearAlgebra; BLAS.set_num_threads(1)` if you observe thread over-subscription.
- **Reproducibility**: SAEM uses random sampling (MH steps). With `nthreads > 1`, subject processing order is non-deterministic, so results may differ slightly between runs. Use `nthreads=1` if exact reproducibility across runs is required.

## Comparing Methods

| Feature | FOCE/FOCE-I | SAEM | ITS | Importance Sampling |
|---------|-------------|------|-----|---------------------|
| Purpose | Parameter estimation | Parameter estimation | Parameter estimation | Uncertainty assessment |
| Called via | `fit()` | `fit_saem()` | `fit_its()` | `importance_sampling()` |
| Speed | Fast | Moderate | Fast | Slow (M full evaluations) |
| Deterministic | Yes | No (stochastic) | Yes | No (Monte Carlo) |
| SEs | Hessian-based | Hessian-based | Hessian-based | Weighted empirical |
| Confidence intervals | Wald (symmetric) | Wald (symmetric) | Wald (symmetric) | Non-parametric (asymmetric) |
| OFV comparable | Yes | Yes (uses FOCE for final OFV) | Yes (uses FOCE for final OFV) | Yes |
| Best for | Most models | Difficult convergence | Warm-start for FOCE, quick estimates | Uncertainty validation |
