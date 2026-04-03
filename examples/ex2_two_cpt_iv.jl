"""
Example: fit a 2-compartment IV bolus PK model to a small simulated dataset.

Run from the package root:
  julia --project examples/run_two_cpt_iv.jl
"""

using JuliaNLME, DataFrames, Random, Printf, TidierPlots

include("utils.jl")

# ---------------------------------------------------------------------------
# True population parameters
# ---------------------------------------------------------------------------

true_theta = [5.0, 15.0, 3.0, 30.0]   # TVCL, TVV1, TVQ, TVV2  (L/h, L, L/h, L)
true_omega  = [0.10, 0.10, 0.10, 0.10] # BSV variances (log-scale)
true_sigma  = [0.01]                    # proportional residual error variance

# ---------------------------------------------------------------------------
# Parse model and simulate 15 subjects, IV bolus 100 mg
# ---------------------------------------------------------------------------

obs_times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0]

model = parse_model_file(joinpath(@__DIR__, "two_cpt_iv.jnlme"))

Random.seed!(123)

true_params = ModelParameters(
    true_theta, [:TVCL, :TVV1, :TVQ, :TVV2],
    OmegaMatrix(true_omega, [:ETA_CL, :ETA_V1, :ETA_Q, :ETA_V2]),
    SigmaMatrix(true_sigma, [:PROP_ERR])
)

df = create_dataset(1:15, 100.0, obs_times)
sim_out = simulate(model, true_params, df)
df[df.EVID .== 0, :DV] = sim_out.dv

println("Simulated $(length(unique(df.ID))) subjects, $(sum(df.EVID .== 0)) observations")

# ---------------------------------------------------------------------------
# Build initial parameters (deliberately offset from truth)
# ---------------------------------------------------------------------------

omega_init = OmegaMatrix([0.15, 0.15, 0.15, 0.15], [:ETA_CL, :ETA_V1, :ETA_Q, :ETA_V2])
init_params = ModelParameters(
    [4.0, 12.0, 2.0, 25.0],   # theta initial values
    [:TVCL, :TVV1, :TVQ, :TVV2],
    omega_init,
    SigmaMatrix([0.02], [:PROP_ERR])
)

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

result = fit(model, df, init_params;
                    outer_maxiter=500,
                    run_covariance_step=true,
                    optimizer = :LD_SLSQP,
                    verbose=true)

print_results(result)

println("\nParameter table:")
display(parameter_table(result))

println("\nTrue values:  TVCL=$(true_theta[1])  TVV1=$(true_theta[2])  TVQ=$(true_theta[3])  TVV2=$(true_theta[4])")

# ---------------------------------------------------------------------------
# Observations table (PRED, IPRED, CWRES, IWRES, ETAs)
# ---------------------------------------------------------------------------

tab = sdtab(result, df)
println("\nObservations table (first 10 rows):")
display(first(tab, 10))

# ---------------------------------------------------------------------------
# Goodness of fit plots
# ---------------------------------------------------------------------------

xy_max = max(maximum(tab.DV), maximum(tab.PRED), maximum(tab.IPRED)) * 1.1
identity_line = DataFrame(x = [0.0, xy_max], y = [0.0, xy_max])

p1 = ggplot(tab, @aes(x = PRED, y = DV)) +
    geom_point(alpha = 0.7) +
    geom_line(identity_line, @aes(x = x, y = y), color = "red", linetype = :dash) +
    labs(x = "Population prediction (PRED)", y = "Observed (DV)", title = "DV vs PRED")

p2 = ggplot(tab, @aes(x = IPRED, y = DV)) +
    geom_point(alpha = 0.7) +
    geom_line(identity_line, @aes(x = x, y = y), color = "red", linetype = :dash) +
    labs(x = "Individual prediction (IPRED)", y = "Observed (DV)", title = "DV vs IPRED")

ggsave(joinpath(@__DIR__, "gof_two_cpt_iv_pred.png"),  p1, width = 500, height = 400)
ggsave(joinpath(@__DIR__, "gof_two_cpt_iv_ipred.png"), p2, width = 500, height = 400)
println("\nGOF plots saved to examples/gof_two_cpt_iv_pred.png and gof_two_cpt_iv_ipred.png")

# ---------------------------------------------------------------------------
# Re-fit with diagonal OMEGA (no between-subject covariances)
# ---------------------------------------------------------------------------
# The full model above estimates a 4×4 lower-triangular OMEGA (10 parameters).
# Setting diagonal=true restricts OMEGA to 4 independent variance terms only,
# which is the appropriate structure when ETAs are assumed uncorrelated.

println("\n\n" * "="^60)
println("  Re-fitting with diagonal OMEGA (4 variance parameters)")
println("="^60)

omega_diag = OmegaMatrix([0.15, 0.15, 0.15, 0.15],
                          [:ETA_CL, :ETA_V1, :ETA_Q, :ETA_V2];
                          diagonal = true)
init_diag = ModelParameters(
    [4.0, 12.0, 2.0, 25.0],
    [:TVCL, :TVV1, :TVQ, :TVV2],
    omega_diag,
    SigmaMatrix([0.02], [:PROP_ERR])
)

result_diag = fit(model, df, init_diag;
                  outer_maxiter=500,
                  run_covariance_step=true,
                  verbose=false)

print_results(result_diag)

println("\nModel comparison:")
@printf "  Full OMEGA  (10 params):  OFV = %.3f   AIC = %.3f\n" result.ofv result.aic
@printf "  Diag OMEGA  ( 4 params):  OFV = %.3f   AIC = %.3f\n" result_diag.ofv result_diag.aic
