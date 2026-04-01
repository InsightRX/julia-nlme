"""
Fit a 2-compartment IV infusion PK model to the UVM dataset.

Matches NONMEM run18 parameterization:
  CL = TVCL * (CRCLf/100)^THETA_CRCL * exp(ETA_CL)
  V1 = TVV1 * (FFM/70)               * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2 * (FFM/70)               * exp(ETA_V2)

  Error:  combined (proportional + additive)
  OMEGA:  full BLOCK(3) — CL/V1/V2 covariance

CRCLf is pre-computed as Cockcroft-Gault with FFM:
  (140-AGE) * FFM * (0.85^SEX) / (72*CREAT)
CRCLf is time-varying (CREAT changes in 74/100 subjects).

NONMEM final estimates (run18):
  TVCL=5.35  TVV1=70.3  TVQ=5.39  TVV2=64.5
  THETA_CRCL=0.865
  PROP_ERR(sigma)=0.0187  ADD_ERR(sigma)=1.147
  OMEGA: diag=[0.071, 0.085, 0.565]

Run from the package root:
  julia --project uvm/fit_uvm.jl
"""

using JuliaNLME, DataFrames, Printf, TidierPlots

data_path  = joinpath(@__DIR__, "data", "nmdata_20230216_1.csv")
model_path = joinpath(@__DIR__, "two_cpt_infusion_crcl_ffm.jnlme")

# ---------------------------------------------------------------------------
# Load data
# Specify covariates explicitly — avoids picking up string columns (PATIENTID, EID)
# and redundant pre-transformed columns.
# CRCLf is time-varying (CREAT changes); FFM/WEIGHT/AGE also vary in some subjects.
# ---------------------------------------------------------------------------

pop = read_data(data_path;
                covariate_columns = [:crclf, :ffm, :weight, :sex, :age])

println("Loaded $(length(pop)) subjects, $(sum(s->length(s.observations), pop.subjects)) observations")

s1 = pop[1]
println("\nSubject 1:")
println("  Time-constant: $(s1.covariates)")
println("  Time-varying:  $(keys(s1.tvcov))")
println("  CRCLf profile: $(round.(get(s1.tvcov, :crclf, Float64[]), digits=1))")

# ---------------------------------------------------------------------------
# Parse model and fit (FOCE-I)
# Initial parameter values, bounds, and omega/sigma starting points are
# read from the [parameters] block of the .jnlme file.
# ---------------------------------------------------------------------------

model = parse_model_file(model_path)

result = fit(model, pop;
             outer_maxiter       = 500,
             interaction         = true,
             run_covariance_step = true,
             verbose             = true,
             global_search  = false,
             global_maxeval = 100,
             n_starts       = 5,
             optimizer      = :bfgs)

#             n_starts            = 5,      # run 5 starts
#             start_jitter        = 0.5)

print_results(result)

println("\nParameter table:")
display(parameter_table(result))

println("\nNONMEM run18 reference:")
println("  TVCL=5.35  TVV1=70.3  TVQ=5.39  TVV2=64.5  THETA_CRCL=0.865")
println("  PROP_ERR(sigma²)=0.0187  ADD_ERR(sigma²)=1.147")
println("  OMEGA diag: CL=0.071  V1=0.085  V2=0.565")

# ---------------------------------------------------------------------------
# OFV comparison: evaluate our FOCEI objective at NONMEM's parameter values.
# If NONMEM's params give a LOWER OFV under our code → optimization failure.
# If they give a HIGHER OFV → the two objectives have different optima (or
# a different dataset/parameterisation).
# ---------------------------------------------------------------------------

println("\n--- OFV at NONMEM's parameters ---")
nm_omega = OmegaMatrix([0.071, 0.085, 0.565], [:ETA_CL, :ETA_V1, :ETA_V2]; diagonal=true)
nm_params = ModelParameters(
    [5.35, 70.3, 5.39, 64.5, 0.865],
    [:TVCL, :TVV1, :TVQ, :TVV2, :THETA_CRCL],
    nm_omega,
    SigmaMatrix([0.0187, 1.147], [:PROP_ERR, :ADD_ERR])
)

nm_eta_hats, nm_H_mats, _ = JuliaNLME.run_inner_loop(pop, nm_params, model;
                                                        maxiter=200, tol=1e-6)

nll_nm = JuliaNLME.foce_population_nll(nm_params, pop, model,
                                         nm_eta_hats, nm_H_mats;
                                         interaction=true)
ofv_nm = 2 * nll_nm   # no n_obs×log(2π) — matches NONMEM's convention
println(@sprintf("  OFV at NONMEM params: %.3f  (NONMEM reports 994.21)", ofv_nm))
println(@sprintf("  OFV at our solution:  %.3f", result.ofv))

if ofv_nm < result.ofv - 1.0
    println("  → NONMEM's params give a LOWER OFV: optimization is suboptimal by $(round(result.ofv - ofv_nm, digits=1)) units")
elseif ofv_nm > result.ofv + 1.0
    println("  → Our solution is BETTER than NONMEM's under our objective")
else
    println("  → Solutions are equivalent (within 1 OFV unit)")
end

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
    geom_point(alpha = 0.6) +
    geom_line(identity_ln, @aes(x = x, y = y), color = "red", linetype = :dash) +
    labs(x = "PRED (mg/L)", y = "DV (mg/L)", title = "DV vs PRED")

p2 = ggplot(tab, @aes(x = IPRED, y = DV)) +
    geom_point(alpha = 0.6) +
    geom_line(identity_ln, @aes(x = x, y = y), color = "red", linetype = :dash) +
    labs(x = "IPRED (mg/L)", y = "DV (mg/L)", title = "DV vs IPRED")

ggsave(joinpath(@__DIR__, "gof_pred.png"),  p1, width = 500, height = 400)
ggsave(joinpath(@__DIR__, "gof_ipred.png"), p2, width = 500, height = 400)

# CWRES vs TIME
zero_ln = DataFrame(x = [minimum(tab.TIME), maximum(tab.TIME)], y = [0.0, 0.0])
p3 = ggplot(tab, @aes(x = TIME, y = CWRES)) +
    geom_point(alpha = 0.6) +
    geom_line(zero_ln, @aes(x = x, y = y), color = "red", linetype = :dash) +
    labs(x = "Time (h)", y = "CWRES", title = "CWRES vs TIME")

ggsave(joinpath(@__DIR__, "cwres_time.png"), p3, width = 500, height = 400)
println("\nPlots saved to uvm/")

# ---------------------------------------------------------------------------
# ETA plots
# ---------------------------------------------------------------------------

sub_df = DataFrame(
    ID        = [s.id for s in pop.subjects],
    ETA_CL    = [result.subjects[i].eta[1] for i in 1:length(pop.subjects)],
    ETA_V1    = [result.subjects[i].eta[2] for i in 1:length(pop.subjects)],
    ETA_V2    = [result.subjects[i].eta[3] for i in 1:length(pop.subjects)],
    CRCLf_base = [haskey(s.tvcov, :crclf) ? s.tvcov[:crclf][1] : s.covariates[:crclf]
                  for s in pop.subjects],
    FFM_base   = [haskey(s.tvcov, :ffm) ? s.tvcov[:ffm][1] : s.covariates[:ffm]
                  for s in pop.subjects],
)

p4 = ggplot(sub_df, @aes(x = CRCLf_base, y = ETA_CL)) +
    geom_point(alpha = 0.8) +
    geom_hline(yintercept = 0, color = "red", linetype = :dash) +
    labs(x = "Baseline CRCLf (mL/min)", y = "ETA_CL", title = "ETA_CL vs CRCLf")

p5 = ggplot(sub_df, @aes(x = FFM_base, y = ETA_V1)) +
    geom_point(alpha = 0.8) +
    geom_hline(yintercept = 0, color = "red", linetype = :dash) +
    labs(x = "Baseline FFM (kg)", y = "ETA_V1", title = "ETA_V1 vs FFM")

ggsave(joinpath(@__DIR__, "eta_cl_vs_crcl.png"), p4, width = 450, height = 380)
ggsave(joinpath(@__DIR__, "eta_v1_vs_ffm.png"),  p5, width = 450, height = 380)
println("ETA plots saved.")
