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

using JuliaNLME, DataFrames, TidierPlots, OrdinaryDiffEq, Random

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
# Simulate a population using OrdinaryDiffEq directly
# ---------------------------------------------------------------------------

function simulate_mm_subject(id, dose, times, TVVMAX, TVKM, TVV, TVKA,
                               ω_VMAX, ω_V, σ_prop, lloq)
    # Individual parameters (log-normal BSV)
    VMAX = TVVMAX * exp(sqrt(ω_VMAX) * randn())
    KM   = TVKM
    V    = TVV   * exp(sqrt(ω_V)    * randn())
    KA   = TVKA

    # ODE system: state = [depot (mg), central (mg/L)]
    function mm_ode!(du, u, p, t)
        depot, central = u
        du[1] = -KA * depot
        du[2] =  KA * depot / V - VMAX * central / (KM + central)
    end

    u0   = [dose, 0.0]   # depot gets full dose; central starts at 0
    tmax = maximum(times)
    prob = ODEProblem(mm_ode!, u0, (0.0, tmax), nothing)
    sol  = solve(prob, Tsit5(); saveat=times, abstol=1e-10, reltol=1e-8)

    rows = []
    # Dose record (EVID=1)
    push!(rows, (ID=id, TIME=0.0, AMT=dose, DV=missing, EVID=1, MDV=1, CMT=1, RATE=0.0))
    for (t, u) in zip(sol.t, sol.u)
        ipred = u[2]   # central compartment concentration (mg/L)
        dv    = ipred * (1 + sqrt(σ_prop) * randn())
        blq   = dv < lloq
        # BLQ observations are flagged MDV=1 (excluded from likelihood) but
        # retained in the dataset so the time grid is complete for plotting.
        push!(rows, (ID=id, TIME=t, AMT=missing,
                     DV   = blq ? lloq : dv,
                     EVID = 0,
                     MDV  = blq ? 1 : 0,
                     CMT  = 2, RATE=0.0))
    end
    return rows
end

all_rows = []
for id in 1:20
    append!(all_rows, simulate_mm_subject(id, dose_amt, obs_times,
                                           true_TVVMAX, true_TVKM, true_TVV, true_TVKA,
                                           true_omega_VMAX, true_omega_V, true_sigma_prop, lloq))
end
df = DataFrame(all_rows)

# ---------------------------------------------------------------------------
# Load dataset and parse model
# ---------------------------------------------------------------------------

pop = read_data(df)
println("Loaded $(length(pop)) subjects, $(sum(s->length(s.observations), pop.subjects)) observations")

model = parse_model_file(joinpath(@__DIR__, "mm_oral.jnlme"))
println("Model: $(model.name)  (pk_model=$(model.pk_model))")

# ---------------------------------------------------------------------------
# Fit using ITS → FOCE-I warm-start
# ---------------------------------------------------------------------------

println("\nStage 1: ITS (fast initialization)...")
its_result = fit_its(model, pop; verbose = true)

println("\nITS estimates:")
print_results(its_result)

println("\nStage 2: FOCE-I warm-started from ITS estimates...")
result = fit(model, pop, its_result;
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

tab = sdtab(result, pop)
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
