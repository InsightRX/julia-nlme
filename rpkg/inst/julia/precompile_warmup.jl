# Precompile warmup script for PackageCompiler.jl
#
# This script is run during sysimage creation to trigger JIT compilation of
# the key JuliaNLME code paths. The resulting native code is baked into the
# sysimage so that the first call in a new session is fast.
#
# Exercises: model parsing, data reading, EBE inner loop (ForwardDiff + BFGS),
# FOCE outer loop (L-BFGS, gradient via ForwardDiff), simulate().

using JuliaNLME, DataFrames, CSV

# ---------------------------------------------------------------------------
# Minimal 1-compartment oral model (analytical, no ODE)
# ---------------------------------------------------------------------------

model_src = """
model WarmupOneCpt

  [parameters]
    theta TVCL(0.1,  0.001, 10.0)
    theta TVV(5.0,   0.1,   100.0)
    theta TVKA(1.0,  0.1,   10.0)

    omega ETA_CL ~ 0.09
    omega ETA_V  ~ 0.04

    sigma PROP_ERR ~ 0.01

  [individual_parameters]
    CL = TVCL * exp(ETA_CL)
    V  = TVV  * exp(ETA_V)
    KA = TVKA

  [structural_model]
    pk one_cpt_oral(cl=CL, v=V, ka=KA)

  [error_model]
    DV ~ proportional(PROP_ERR)

end
"""

mfile = tempname() * ".jnlme"
write(mfile, model_src)
model = JuliaNLME.parse_model_file(mfile)
rm(mfile)

# ---------------------------------------------------------------------------
# Tiny dataset: 4 subjects, 5 observations each (enough to warm up all loops)
# ---------------------------------------------------------------------------

rows = []
for id in 1:4
    push!(rows, (ID=id, TIME=0.0,  AMT=100.0,   DV=missing, EVID=1, MDV=1, CMT=1, RATE=0.0))
    for t in [1.0, 4.0, 8.0, 24.0, 48.0]
        push!(rows, (ID=id, TIME=t, AMT=missing, DV=1.0,    EVID=0, MDV=0, CMT=1, RATE=0.0))
    end
end
df = DataFrame(rows)

# ---------------------------------------------------------------------------
# FOCE fit — a few outer iterations to trigger ForwardDiff + BFGS JIT
# ---------------------------------------------------------------------------

result = JuliaNLME.fit(model, df;
    outer_maxiter       = 5,
    run_covariance_step = false,
    verbose             = false)

# ---------------------------------------------------------------------------
# FOCE-I (interaction = true uses a different code path in the likelihood)
# ---------------------------------------------------------------------------

JuliaNLME.fit(model, df;
    outer_maxiter       = 3,
    interaction         = true,
    run_covariance_step = false,
    verbose             = false)

# ---------------------------------------------------------------------------
# ITS and SAEM (each has a distinct M-step / sampling code path)
# ---------------------------------------------------------------------------

JuliaNLME.fit_its(model, df;
    n_iter              = 5,
    run_covariance_step = false,
    verbose             = false)

JuliaNLME.fit_saem(model, df;
    n_iter_exploration  = 5,
    n_iter_convergence  = 5,
    run_covariance_step = false,
    verbose             = false)

# ---------------------------------------------------------------------------
# Simulate (exercises the prediction + noise sampling path)
# ---------------------------------------------------------------------------

JuliaNLME.simulate(model, result, df; n_sims = 2)
