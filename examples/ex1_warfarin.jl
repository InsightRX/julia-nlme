"""
Example: fit a 1-compartment oral PK model to a small simulated warfarin dataset.

Run from the package root:
  julia --project examples/run_warfarin.jl
"""

using JuliaNLME, DataFrames, Random, TidierPlots

include("utils.jl")

# ---------------------------------------------------------------------------
# True population parameters
# ---------------------------------------------------------------------------

true_theta = [0.134, 8.1, 1.0]   # TVCL, TVV, TVKA
true_omega  = [0.07, 0.02, 0.40]  # BSV variances
true_sigma  = [0.01]              # proportional error variance

obs_times = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0]

# ---------------------------------------------------------------------------
# Parse model and simulate a small dataset (10 subjects, oral single dose 100 mg)
# ---------------------------------------------------------------------------

model = parse_model_file(joinpath(@__DIR__, "warfarin_oral.jnlme"))

Random.seed!(42)

true_params = ModelParameters(
    true_theta, [:TVCL, :TVV, :TVKA],
    OmegaMatrix(true_omega, [:ETA_CL, :ETA_V, :ETA_KA]),
    SigmaMatrix(true_sigma, [:PROP_ERR])
)

df = create_dataset(1:10, 100.0, obs_times)
sim_out = simulate(model, true_params, df)
df[df.EVID .== 0, :DV] = sim_out.dv

println("Simulated $(length(unique(df.ID))) subjects, $(sum(df.EVID .== 0)) observations")

# ---------------------------------------------------------------------------
# Build initial parameters
# ---------------------------------------------------------------------------

omega_init = OmegaMatrix([0.09, 0.04, 0.30], [:ETA_CL, :ETA_V, :ETA_KA])
init_params = ModelParameters(
    [0.2, 10.0, 1.5],   # theta initial values
    [:TVCL, :TVV, :TVKA],
    omega_init,
    SigmaMatrix([0.02], [:PROP_ERR])
)

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

result = fit(model, df, init_params;
             outer_maxiter=300,
             run_covariance_step=true,
             optimizer = :LD_SLSQP,
             verbose=true)

print_results(result)

println("\nParameter table:")
display(parameter_table(result))

println("\nTrue values:  TVCL=$(true_theta[1])  TVV=$(true_theta[2])  TVKA=$(true_theta[3])")

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

ggsave(joinpath(@__DIR__, "gof_warfarin_pred.png"),  p1, width = 500, height = 400)
ggsave(joinpath(@__DIR__, "gof_warfarin_ipred.png"), p2, width = 500, height = 400)
println("\nGOF plots saved to examples/gof_warfarin_pred.png and gof_warfarin_ipred.png")
