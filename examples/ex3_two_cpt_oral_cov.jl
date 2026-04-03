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

include("utils.jl")

# ---------------------------------------------------------------------------
# True population parameters
# ---------------------------------------------------------------------------

true_theta = [5.0, 50.0, 10.0, 100.0, 1.2, 0.75, 0.50]
#              TVCL TVV1  TVQ   TVV2   TVKA WT    CRCL

true_omega = [0.10, 0.10, 0.05, 0.05, 0.15]   # ETA_CL, V1, Q, V2, KA
true_sigma = [0.02]

# ---------------------------------------------------------------------------
# Parse model and simulate 30 subjects — WT constant, CRCL declines over time
# ---------------------------------------------------------------------------
#
# CRCL(t) = max(crcl_0 - rate * t, 20)
# rate ~ Uniform(0, 0.5) mL/min/h  →  max 24 mL/min drop over 48 h
#
# Concentrations are simulated using the CRCL value at each observation time,
# consistent with the time-varying covariate approximation used by FOCE.

obs_times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0]

model = parse_model_file(joinpath(@__DIR__, "two_cpt_oral_cov.jnlme"))

Random.seed!(456)

true_params = ModelParameters(
    true_theta, [:TVCL, :TVV1, :TVQ, :TVV2, :TVKA, :THETA_WT, :THETA_CRCL],
    OmegaMatrix(true_omega, [:ETA_CL, :ETA_V1, :ETA_Q, :ETA_V2, :ETA_KA]),
    SigmaMatrix(true_sigma, [:PROP_ERR])
)

df = create_dataset(1:30, 250.0, obs_times)
df[!, :WT]   = zeros(nrow(df))
df[!, :CRCL] = zeros(nrow(df))

for id in 1:30
    wt        = clamp(70.0 + 15.0 * randn(), 45.0, 120.0)
    crcl_0    = clamp(90.0 + 25.0 * randn(), 30.0, 150.0)
    crcl_rate = rand() * 0.5     # 0–0.5 mL/min per hour
    id_mask = df.ID .== id
    df[id_mask .& (df.EVID .== 1), :WT]   .= wt
    df[id_mask .& (df.EVID .== 1), :CRCL] .= crcl_0
    for (j, t) in enumerate(obs_times)
        row = findfirst(id_mask .& (df.EVID .== 0) .& (df.TIME .== t))
        df[row, :WT]   = wt
        df[row, :CRCL] = max(crcl_0 - crcl_rate * t, 20.0)
    end
end

sim_out = simulate(model, true_params, df)
df[df.EVID .== 0, :DV] = sim_out.dv

# ---------------------------------------------------------------------------
# Load dataset
# WT is constant within subjects  → stored in subject.covariates
# CRCL varies within subjects     → stored in subject.tvcov (detected automatically)
# ---------------------------------------------------------------------------

println("Simulated $(length(unique(df.ID))) subjects, $(sum(df.EVID .== 0)) observations")
s1_obs = filter(r -> r.ID == 1 && r.EVID == 0, df)
println("Subject 1 WT=$(round(s1_obs.WT[1], digits=1))  CRCL over time: $(round.(s1_obs.CRCL, digits=1))")

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
# Fit
# ---------------------------------------------------------------------------

result = fit(model, df, init_params;
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

tab = sdtab(result, df)
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
obs_df = filter(:EVID => ==(0), df)
crcl_df = obs_df[!, [:ID, :TIME, :CRCL]]
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

subj_first = combine(groupby(obs_df, :ID), first)
sub_df = DataFrame(
    ID        = subj_first.ID,
    ETA_CL    = [result.subjects[i].eta[1] for i in 1:nrow(subj_first)],
    WT        = subj_first.WT,
    CRCL_base = subj_first.CRCL,   # CRCL at first obs ≈ baseline
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
