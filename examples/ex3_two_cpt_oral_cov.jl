"""
Example: fit a 2-compartment oral PK model with weight (WT, time-constant) and
creatinine clearance (CRCL, time-varying) covariates.

CRCL declines linearly over the 48-hour study in each subject, simulating a
mild acute kidney injury scenario.  The datareader detects CRCL as time-varying
because its value changes within subjects; WT remains constant (absorbed into
the time-constant covariate slot).

Covariate model:
  CL = TVCL * (WT/70)^THETA_WT * (CRCL/100)^THETA_CRCL * exp(ETA_CL)
  V1 = TVV1 * (WT/70)^THETA_WT * exp(ETA_V1)

Run from the package root:
  julia --project examples/ex3_two_cpt_oral_cov.jl
"""

using JuliaNLME, DataFrames, Random, Printf, TidierPlots

# ---------------------------------------------------------------------------
# True population parameters
# ---------------------------------------------------------------------------

true_theta = [5.0, 50.0, 10.0, 100.0, 1.2, 0.75, 0.50]
#              TVCL TVV1  TVQ   TVV2   TVKA WT    CRCL

true_omega = [0.10, 0.10, 0.05, 0.05, 0.15]   # ETA_CL, V1, Q, V2, KA
true_sigma = [0.02]

# ---------------------------------------------------------------------------
# Simulate 30 subjects — WT constant, CRCL declines over time
# ---------------------------------------------------------------------------
#
# CRCL(t) = max(crcl_0 - rate * t, 20)
# rate ~ Uniform(0, 0.5) mL/min/h  →  max 24 mL/min drop over 48 h
#
# Concentrations are simulated using the CRCL value at each observation time,
# consistent with the time-varying covariate approximation used by FOCE.

obs_times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0]

Random.seed!(456)

function simulate_subject(id, dose, times, θ, ω_var, σ_var, wt, crcl_0, crcl_rate)
    tvcl, tvv1, tvq, tvv2, tvka, θ_wt, θ_crcl = θ
    eta_cl = sqrt(ω_var[1]) * randn()
    v1 = tvv1 * (wt/70)^θ_wt * exp(sqrt(ω_var[2]) * randn())
    q  = tvq  * exp(sqrt(ω_var[3]) * randn())
    v2 = tvv2 * exp(sqrt(ω_var[4]) * randn())
    ka = tvka * exp(sqrt(ω_var[5]) * randn())
    rows = []
    # Dose record: CRCL at baseline
    push!(rows, (ID=id, TIME=0.0, AMT=dose, DV=missing, EVID=1, MDV=1, CMT=1, RATE=0.0,
                 WT=wt, CRCL=crcl_0))
    for t in times
        crcl_t = max(crcl_0 - crcl_rate * t, 20.0)
        cl_t   = tvcl * (wt/70)^θ_wt * (crcl_t/100)^θ_crcl * exp(eta_cl)
        ipred  = two_cpt_oral(; cl=cl_t, v1=v1, q=q, v2=v2, ka=ka, dose=dose, t=t)
        dv     = ipred * (1 + sqrt(σ_var[1]) * randn())
        push!(rows, (ID=id, TIME=t, AMT=missing, DV=max(dv, 0.001), EVID=0, MDV=0, CMT=1, RATE=0.0,
                     WT=wt, CRCL=crcl_t))
    end
    return rows
end

all_rows = []
for id in 1:30
    wt        = clamp(70.0 + 15.0 * randn(), 45.0, 120.0)
    crcl_0    = clamp(90.0 + 25.0 * randn(), 30.0, 150.0)
    crcl_rate = rand() * 0.5     # 0–0.5 mL/min per hour
    append!(all_rows, simulate_subject(id, 250.0, obs_times,
                                       true_theta, true_omega, true_sigma,
                                       wt, crcl_0, crcl_rate))
end
df = DataFrame(all_rows)

# ---------------------------------------------------------------------------
# Load dataset
# WT is constant within subjects  → stored in subject.covariates
# CRCL varies within subjects     → stored in subject.tvcov (detected automatically)
# ---------------------------------------------------------------------------

pop = read_data(df)
println("Loaded $(length(pop)) subjects, $(sum(s->length(s.observations), pop.subjects)) observations")

s1 = pop[1]
println("Subject 1 — time-constant covariates: $(keys(s1.covariates))")
println("Subject 1 — time-varying  covariates: $(keys(s1.tvcov))")
println("Subject 1 CRCL over time: $(round.(s1.tvcov[:crcl], digits=1))")

# ---------------------------------------------------------------------------
# Build initial parameters (deliberately offset from truth)
# ---------------------------------------------------------------------------

omega_init  = OmegaMatrix([0.15, 0.15, 0.08, 0.08, 0.20],
                           [:ETA_CL, :ETA_V1, :ETA_Q, :ETA_V2, :ETA_KA];
                           diagonal = true)
init_params = ModelParameters(
    [4.0, 40.0, 8.0, 80.0, 1.0, 0.6, 0.3],
    [:TVCL, :TVV1, :TVQ, :TVV2, :TVKA, :THETA_WT, :THETA_CRCL],
    omega_init,
    SigmaMatrix([0.04], [:PROP_ERR])
)

# ---------------------------------------------------------------------------
# Parse model and fit
# ---------------------------------------------------------------------------

model = parse_model_file(joinpath(@__DIR__, "two_cpt_oral_cov.jnlme"))

result = fit(model, pop, init_params;
             outer_maxiter = 500,
             run_covariance_step = true,
             interaction = true,
             optimizer = :bfgs,
             verbose = true)

print_results(result)

println("\nParameter table:")
display(parameter_table(result))

println("\nTrue values:")
@printf "  TVCL=%.2f  TVV1=%.1f  TVQ=%.1f  TVV2=%.1f  TVKA=%.2f\n" true_theta[1:5]...
@printf "  THETA_WT=%.2f  THETA_CRCL=%.2f\n" true_theta[6] true_theta[7]

# ---------------------------------------------------------------------------
# Observations table
# ---------------------------------------------------------------------------

tab = sdtab(result, pop)
println("\nObservations table (first 10 rows):")
display(first(tab, 10))

# ---------------------------------------------------------------------------
# Goodness of fit plots
# ---------------------------------------------------------------------------

xy_max      = max(maximum(tab.DV), maximum(tab.PRED), maximum(tab.IPRED)) * 1.1
identity_ln = DataFrame(x = [0.0, xy_max], y = [0.0, xy_max])

p1 = ggplot(tab, @aes(x = PRED, y = DV)) +
    geom_point(alpha = 0.7) +
    geom_line(identity_ln, @aes(x = x, y = y), color = "red", linetype = :dash) +
    labs(x = "Population prediction (PRED)", y = "Observed (DV)", title = "DV vs PRED")

p2 = ggplot(tab, @aes(x = IPRED, y = DV)) +
    geom_point(alpha = 0.7) +
    geom_line(identity_ln, @aes(x = x, y = y), color = "red", linetype = :dash) +
    labs(x = "Individual prediction (IPRED)", y = "Observed (DV)", title = "DV vs IPRED")

ggsave(joinpath(@__DIR__, "gof_two_cpt_oral_cov_pred.png"),  p1, width = 500, height = 400)
ggsave(joinpath(@__DIR__, "gof_two_cpt_oral_cov_ipred.png"), p2, width = 500, height = 400)
println("\nGOF plots saved.")

# ---------------------------------------------------------------------------
# Covariate profile plot: CRCL over time for all subjects
# ---------------------------------------------------------------------------

# Build a long-format DataFrame of CRCL observations per subject per time
crcl_rows = [(ID=s.id, TIME=s.obs_times[j], CRCL=s.tvcov[:crcl][j])
             for s in pop.subjects for j in eachindex(s.obs_times)]
crcl_df = DataFrame(crcl_rows)
crcl_df[!, :ID_str] = string.(crcl_df.ID)   # for colour grouping

p_crcl = ggplot(crcl_df, @aes(x = TIME, y = CRCL, color = ID_str)) +
    geom_line(alpha = 0.6) +
    labs(x = "Time (h)", y = "CRCL (mL/min)", title = "CRCL profiles over time") +
    theme_minimal()

ggsave(joinpath(@__DIR__, "crcl_profiles.png"), p_crcl, width = 550, height = 380)
println("CRCL profile plot saved to examples/crcl_profiles.png")

# ---------------------------------------------------------------------------
# Covariate-ETA plots: use baseline CRCL (first obs time ≈ t=0.5) and WT
# ---------------------------------------------------------------------------

sub_df = DataFrame(
    ID          = [s.id for s in pop.subjects],
    ETA_CL      = [result.subjects[i].eta[1] for i in 1:length(pop.subjects)],
    WT          = [s.covariates[:wt]      for s in pop.subjects],
    CRCL_base   = [s.tvcov[:crcl][1]     for s in pop.subjects],   # CRCL at first obs ≈ baseline
)

p3 = ggplot(sub_df, @aes(x = WT, y = ETA_CL)) +
    geom_point(alpha = 0.8) +
    geom_hline(yintercept = 0, color = "red", linetype = :dash) +
    labs(x = "Weight (kg)", y = "ETA_CL", title = "ETA_CL vs WT")

p4 = ggplot(sub_df, @aes(x = CRCL_base, y = ETA_CL)) +
    geom_point(alpha = 0.8) +
    geom_hline(yintercept = 0, color = "red", linetype = :dash) +
    labs(x = "Baseline CRCL (mL/min)", y = "ETA_CL", title = "ETA_CL vs baseline CRCL")

ggsave(joinpath(@__DIR__, "eta_cl_vs_wt.png"),        p3, width = 450, height = 380)
ggsave(joinpath(@__DIR__, "eta_cl_vs_crcl_base.png"), p4, width = 450, height = 380)
println("Covariate-ETA plots saved.")
