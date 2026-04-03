"""
Example 4: 1-compartment oral PK model with Michaelis-Menten (saturable) elimination.

This example demonstrates ODE-based estimation in JuliaNLME. Unlike examples 1–3
which use analytical solutions, this model is defined with explicit differential
equations in `mm_oral.jnlme`:

  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V  -  VMAX * central / (KM + central)

At high concentrations (C >> KM), the elimination rate approaches VMAX and is
no longer proportional to concentration. This gives a convex terminal slope on
a log-concentration plot — distinctly non-linear compared to first-order models.

Run from the package root:
  julia --project=. examples/ex4_ode_mm.jl
"""

using JuliaNLME, DataFrames, TidierPlots, Random

include("utils.jl")

Random.seed!(1234)

# ---------------------------------------------------------------------------
# True parameter values
# ---------------------------------------------------------------------------

true_TVVMAX = 4.0    # maximum elimination rate (mg/h)
true_TVKM   = 6.0    # Michaelis-Menten constant (mg/L)
true_TVV    = 12.0   # volume of distribution (L)
true_TVKA   = 1.5    # first-order absorption rate (1/h)

true_omega_VMAX = 0.15   # variance on VMAX (BSV ~40% CV)
true_omega_V    = 0.10   # variance on V    (~32% CV)

true_sigma_prop = 0.02   # proportional residual error variance (14% CV)

lloq      = 0.01    # lower limit of quantification (mg/L); observations below LLOQ are excluded (MDV=1)
obs_times = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0]
dose_amt  = 200.0   # mg, single oral dose

# ---------------------------------------------------------------------------
# Parse model and simulate a population
# ---------------------------------------------------------------------------

model = parse_model_file(joinpath(@__DIR__, "mm_oral.jnlme"))

true_params = ModelParameters(
    [true_TVVMAX, true_TVKM, true_TVV, true_TVKA], [:TVVMAX, :TVKM, :TVV, :TVKA],
    OmegaMatrix([true_omega_VMAX, true_omega_V], [:ETA_VMAX, :ETA_V]),
    SigmaMatrix([true_sigma_prop], [:PROP_ERR])
)

df = create_dataset(1:20, dose_amt, obs_times; cmt=2)
sim_out = simulate(model, true_params, df)

# Apply BLQ: observations below LLOQ are flagged MDV=1 but retained for plotting
obs_idx = findall(df.EVID .== 0)
for (i, idx) in enumerate(obs_idx)
    dv_i = sim_out.dv[i]
    df[idx, :DV]  = dv_i < lloq ? lloq : dv_i
    df[idx, :MDV] = dv_i < lloq ? 1    : 0
end

println("Simulated $(length(unique(df.ID))) subjects, $(sum(df.EVID .== 0)) observations")
println("Model: $(model.name)  (pk_model=$(model.pk_model))")

# ---------------------------------------------------------------------------
# Fit using ITS → FOCE-I warm-start
# ---------------------------------------------------------------------------

println("\nStage 1: ITS (fast initialization)...")
its_result = fit_its(model, df; verbose = true)

println("\nITS estimates:")
print_results(its_result)

println("\nStage 2: FOCE-I warm-started from ITS estimates...")
result = fit(model, df, its_result;
             interaction         = true,
             outer_maxiter       = 400,
             run_covariance_step = true,
             verbose             = true)

print_results(result)

println("\nParameter table:")
display(parameter_table(result))

println("\nTrue values:")
println("  TVVMAX = $true_TVVMAX  TVKM = $true_TVKM  TVV = $true_TVV  TVKA = $true_TVKA")

# ---------------------------------------------------------------------------
# Observations table
# ---------------------------------------------------------------------------

tab = sdtab(result, df)
println("\nObservations table (first 10 rows):")
display(first(tab, 10))

# ---------------------------------------------------------------------------
# Goodness-of-fit plots
# ---------------------------------------------------------------------------

xy_max = max(maximum(tab.DV), maximum(tab.PRED), maximum(tab.IPRED)) * 1.1
identity_line = DataFrame(x=[0.0, xy_max], y=[0.0, xy_max])

p1 = ggplot(tab, @aes(x=PRED, y=DV)) +
    geom_point(alpha=0.6) +
    geom_line(identity_line, @aes(x=x, y=y), color="red", linetype=:dash) +
    labs(x="Population prediction (PRED)", y="Observed (DV)",
         title="MM oral — DV vs PRED")

p2 = ggplot(tab, @aes(x=IPRED, y=DV)) +
    geom_point(alpha=0.6) +
    geom_line(identity_line, @aes(x=x, y=y), color="red", linetype=:dash) +
    labs(x="Individual prediction (IPRED)", y="Observed (DV)",
         title="MM oral — DV vs IPRED")

# Individual concentration–time profile for first subject
sub1 = filter(:ID => ==(1), tab)
p3 = ggplot(sub1, @aes(x=TIME, y=DV)) +
    geom_point(size=2) +
    geom_line(@aes(y=IPRED), color="steelblue") +
    geom_line(@aes(y=PRED),  color="red", linetype=:dash) +
    labs(x="Time (h)", y="Concentration (mg/L)",
         title="Subject 1: observed (●), IPRED (—), PRED (- -)")

ggsave(joinpath(@__DIR__, "gof_mm_oral_pred.png"),   p1, width=500, height=400)
ggsave(joinpath(@__DIR__, "gof_mm_oral_ipred.png"),  p2, width=500, height=400)
ggsave(joinpath(@__DIR__, "gof_mm_oral_conc.png"),   p3, width=600, height=400)
println("\nGOF plots saved to examples/gof_mm_oral_*.png")
