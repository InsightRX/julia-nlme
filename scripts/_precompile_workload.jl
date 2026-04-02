"""
Precompile workload for PackageCompiler.
Exercises the full fit() call stack on a tiny synthetic dataset so that
all hot-path methods are compiled into the sysimage.
"""

using JuliaNLME, ArgParse, Printf, TOML

# Minimal one-compartment model
model_str = """
model PrecompileModel
  [parameters]
    theta TVCL(2.0, 0.01, 100.0)
    theta TVV(20.0, 0.1, 500.0)
    omega ETA_CL ~ 0.09
    sigma PROP_ERR ~ 0.01
  [individual_parameters]
    CL = TVCL * exp(ETA_CL)
    V  = TVV
  [structural_model]
    pk one_cpt_iv_bolus(cl=CL, v=V)
  [error_model]
    DV ~ proportional(PROP_ERR)
end
"""

model = parse_model_string(model_str)

using DataFrames
df = DataFrame(
    ID   = [1, 1, 1, 2, 2, 2],
    TIME = [0.0, 4.0, 24.0, 0.0, 4.0, 24.0],
    AMT  = [100.0, missing, missing, 80.0, missing, missing],
    DV   = [missing, 3.0, 1.0, missing, 2.5, 0.8],
    EVID = [1, 0, 0, 1, 0, 0],
    MDV  = [1, 0, 0, 1, 0, 0],
)
pop = read_data(df)

result = fit(model, pop;
    outer_maxiter = 10,
    run_covariance_step = false,
    verbose = false)

print_results(result)
parameter_table(result)
sdtab(result, pop)
