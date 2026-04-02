using Test, JuliaNLME, LinearAlgebra, ForwardDiff

# ---------------------------------------------------------------------------
# PK equations
# ---------------------------------------------------------------------------

@testset "1-CMT IV bolus" begin
    cl, v, dose, t = 2.0, 10.0, 100.0, 2.0
    k = cl / v
    expected = (dose/v) * exp(-k * t)
    @test one_cpt_iv_bolus(; cl, v, dose, t) ≈ expected

    # Zero time
    @test one_cpt_iv_bolus(; cl, v, dose, t=0.0) ≈ dose/v

    # AD compatibility
    g = ForwardDiff.gradient(p -> one_cpt_iv_bolus(; cl=p[1], v=p[2], dose=100.0, t=2.0),
                             [cl, v])
    @test all(isfinite, g)
end

@testset "1-CMT oral" begin
    cl, v, ka, f, dose, t = 2.0, 20.0, 1.5, 1.0, 100.0, 4.0
    k = cl / v
    expected = (f*dose*ka) / (v*(ka-k)) * (exp(-k*t) - exp(-ka*t))
    @test one_cpt_oral(; cl, v, ka=ka, f=f, dose, t) ≈ expected rtol=1e-8

    # Singularity: ka ≈ k
    ka_sing = cl/v + 1e-8
    c_sing = one_cpt_oral(; cl, v, ka=ka_sing, dose, t)
    c_limit = (dose*ka_sing/v) * t * exp(-(cl/v)*t)  # L'Hopital limit
    @test c_sing ≈ c_limit rtol=1e-3

    # AD compatible
    g = ForwardDiff.gradient(p -> one_cpt_oral(; cl=p[1], v=p[2], ka=p[3], dose=100.0, t=4.0),
                             [cl, v, ka])
    @test all(isfinite, g)
end

@testset "1-CMT infusion" begin
    cl, v, dose, dur = 2.0, 10.0, 100.0, 1.0
    t_during = 0.5
    rate = dose / dur
    k = cl / v
    # During infusion
    @test one_cpt_infusion(; cl, v, dose, duration=dur, t=t_during) ≈
          (rate/cl) * (1 - exp(-k*t_during)) rtol=1e-8
    # After infusion
    t_post = 2.0
    c_eoi  = (rate/cl) * (1 - exp(-k*dur))
    @test one_cpt_infusion(; cl, v, dose, duration=dur, t=t_post) ≈
          c_eoi * exp(-k*(t_post - dur)) rtol=1e-8
end

@testset "2-CMT IV bolus" begin
    cl, v1, q, v2, dose, t = 5.0, 10.0, 2.0, 20.0, 100.0, 3.0
    k10 = cl/v1; k12 = q/v1; k21 = q/v2
    s = k10+k12+k21
    disc = sqrt(s^2 - 4*k10*k21)
    α = (s+disc)/2; β = (s-disc)/2
    A = (dose/v1) * (α-k21)/(α-β)
    B = (dose/v1) * (k21-β)/(α-β)
    @test two_cpt_iv_bolus(; cl, v1, q, v2, dose, t) ≈ A*exp(-α*t)+B*exp(-β*t) rtol=1e-8

    # AD
    g = ForwardDiff.gradient(p -> two_cpt_iv_bolus(; cl=p[1], v1=p[2], q=p[3], v2=p[4],
                                                     dose=100.0, t=3.0),
                             [cl, v1, q, v2])
    @test all(isfinite, g)
end

# ---------------------------------------------------------------------------
# Multi-dose superposition
# ---------------------------------------------------------------------------

@testset "Multi-dose superposition" begin
    doses = [DoseEvent(0.0, 100.0; cmt=1),
             DoseEvent(24.0, 100.0; cmt=1)]
    params = (cl=2.0, v=20.0, ka=1.0)

    subject = Subject(1, doses, [12.0, 36.0], [0.0, 0.0], [1, 1], Dict{Symbol,Float64}())
    preds = predict_subject(:one_cpt_oral, params, subject)
    @test length(preds) == 2
    @test all(preds .>= 0)

    # At t=36: sum of dose1 (at τ=36) and dose2 (at τ=12)
    c1 = one_cpt_oral(; cl=2.0, v=20.0, ka=1.0, dose=100.0, t=36.0)
    c2 = one_cpt_oral(; cl=2.0, v=20.0, ka=1.0, dose=100.0, t=12.0)
    @test preds[2] ≈ c1 + c2 rtol=1e-8
end

# ---------------------------------------------------------------------------
# Data reader
# ---------------------------------------------------------------------------

@testset "Data reader" begin
    using DataFrames

    df = DataFrame(
        ID    = [1, 1, 1, 2, 2, 2],
        TIME  = [0.0, 1.0, 4.0, 0.0, 2.0, 8.0],
        AMT   = [100.0, missing, missing, 100.0, missing, missing],
        DV    = [missing, 1.5, 0.8, missing, 2.1, 1.0],
        EVID  = [1, 0, 0, 1, 0, 0],
        MDV   = [1, 0, 0, 1, 0, 0],
        WT    = [70.0, 70.0, 70.0, 85.0, 85.0, 85.0]
    )

    pop = read_data(df; covariate_columns=[:wt])
    @test length(pop) == 2
    @test pop[1].id == 1
    @test length(pop[1].doses) == 1
    @test pop[1].doses[1].amt == 100.0
    @test length(pop[1].observations) == 2
    @test pop[1].observations ≈ [1.5, 0.8]
    @test pop[1].covariates[:wt] ≈ 70.0
    @test pop[2].covariates[:wt] ≈ 85.0
end

# ---------------------------------------------------------------------------
# OmegaMatrix Cholesky
# ---------------------------------------------------------------------------

@testset "OmegaMatrix" begin
    ω = OmegaMatrix([0.09, 0.04], [:ETA_CL, :ETA_V])
    @test ω.matrix ≈ Diagonal([0.09, 0.04]) atol=1e-12
    @test n_etas(ω) == 2
    @test ω.chol * ω.chol' ≈ ω.matrix atol=1e-12
end

# ---------------------------------------------------------------------------
# Residual error models
# ---------------------------------------------------------------------------

@testset "Residual error models" begin
    σ_add  = [0.01]
    σ_prop = [0.04]
    σ_comb = [0.04, 0.01]

    @test residual_variance(:additive, 5.0, σ_add) ≈ 0.01
    @test residual_variance(:proportional, 5.0, σ_prop) ≈ (5.0 * sqrt(0.04))^2 rtol=1e-8
    @test residual_variance(:combined, 5.0, σ_comb) ≈ (5.0*sqrt(0.04))^2 + 0.01 rtol=1e-8
end

# ---------------------------------------------------------------------------
# Parameterization round-trip
# ---------------------------------------------------------------------------

@testset "Parameter pack/unpack round-trip" begin
    omega = OmegaMatrix([0.09, 0.04], [:ETA_CL, :ETA_V])
    params = ModelParameters(
        [2.0, 30.0, 1.0],
        [:TVCL, :TVV, :TVKA],
        omega,
        SigmaMatrix([0.01], [:PROP_ERR])
    )

    x = pack_params(params)
    params2 = unpack_params(x, params)

    @test params2.theta ≈ params.theta rtol=1e-10
    @test params2.omega.matrix ≈ params.omega.matrix rtol=1e-10
    @test params2.sigma.values ≈ params.sigma.values rtol=1e-10
end

# ---------------------------------------------------------------------------
# Model parser (smoke test)
# ---------------------------------------------------------------------------

@testset "Model parser" begin
    model_str = """
    model TestModel

      [parameters]
        theta TVCL(2.0, 0.01, 100.0)
        theta TVV(20.0, 0.1, 500.0)
        theta TVKA(1.0, 0.01, 10.0)
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
    model = parse_model_string(model_str)
    @test model.name == "TestModel"
    @test model.pk_model == :one_cpt_oral
    @test model.error_model == :proportional
    @test model.n_theta == 3
    @test model.n_eta == 2
    @test model.n_epsilon == 1

    # Test the generated pk_param_fn
    theta = [2.0, 20.0, 1.0]
    eta   = [0.1, -0.1]
    cov   = Dict{Symbol,Float64}()
    pk = model.pk_param_fn(theta, eta, cov)
    @test pk.cl ≈ 2.0 * exp(0.1) rtol=1e-8
    @test pk.v  ≈ 20.0 * exp(-0.1) rtol=1e-8
    @test pk.ka ≈ 1.0 rtol=1e-8
end

@testset "Fixed parameters: bounds pinning" begin
    # A model with one fixed sigma — verifies that ε-pinning keeps bounds valid
    # and that the optimizer does not immediately terminate (≥2 iterations expected).
    model_str = """
    model FixedSigmaModel

      [parameters]
        theta TVCL(2.0, 0.01, 100.0)
        theta TVV(20.0, 0.1, 500.0)
        omega ETA_CL ~ 0.09
        sigma PROP_ERR ~ 0.05
        sigma ADD_ERR  ~ 0.25 fix

      [individual_parameters]
        CL = TVCL * exp(ETA_CL)
        V  = TVV

      [structural_model]
        pk one_cpt_iv_bolus(cl=CL, v=V)

      [error_model]
        DV ~ combined(PROP_ERR, ADD_ERR)

    end
    """
    model = parse_model_string(model_str)
    @test model.default_params.packed_fixed[end] == true    # ADD_ERR sigma is fixed
    @test model.default_params.packed_fixed[end-1] == false # PROP_ERR sigma is free

    # Fixed params are excluded from the optimizer — verify the packed_fixed
    # index correctly identifies ADD_ERR as the last sigma.
    x0, lower, upper = JuliaNLME.initial_packed(model.default_params)
    # All bounds should be strictly ordered (no pinning needed now)
    @test all(lower .< upper)
    # The free (non-fixed) bounds should cover a reasonable range
    free_idx = findall(.!model.default_params.packed_fixed)
    @test all(upper[free_idx] .- lower[free_idx] .> 0.1)
end

# ---------------------------------------------------------------------------
# Time-varying covariates
# ---------------------------------------------------------------------------

@testset "Time-varying covariates: detection and LOCF" begin
    using DataFrames

    # Subject 1: WT changes between observations → time-varying
    # Subject 2: WT constant                     → still TV (dataset-level classification)
    df = DataFrame(
        ID   = [1,   1,   1,   1,   2,   2,   2],
        TIME = [0.0, 4.0, 24.0, 48.0, 0.0, 12.0, 48.0],
        AMT  = [100.0, missing, missing, missing, 100.0, missing, missing],
        DV   = [missing, 3.0, 1.5, 0.8, missing, 2.0, 0.9],
        EVID = [1, 0, 0, 0, 1, 0, 0],
        MDV  = [1, 0, 0, 0, 1, 0, 0],
        WT   = [70.0, 70.0, 65.0, 65.0, 80.0, 80.0, 80.0],  # WT changes in subj 1
        AGE  = [40.0, 40.0, 40.0, 40.0, 55.0, 55.0, 55.0],  # AGE constant for all
    )

    pop = read_data(df)

    # AGE should be time-constant; WT should be time-varying
    s1 = pop[1]
    @test haskey(s1.covariates, :age)   # constant → in covariates
    @test !haskey(s1.tvcov, :age)
    @test haskey(s1.tvcov, :wt)         # varying  → in tvcov
    @test !haskey(s1.covariates, :wt)

    # Subject 1: 3 observations, tvcov[:wt] should align with LOCF
    @test length(s1.tvcov[:wt]) == 3
    @test s1.tvcov[:wt] ≈ [70.0, 65.0, 65.0]  # LOCF: 70@t4, 65@t24, 65@t48

    # Subject 2: WT constant → tvcov entry still present but all same value
    s2 = pop[2]
    @test length(s2.tvcov[:wt]) == 2
    @test all(s2.tvcov[:wt] .≈ 80.0)
end

@testset "Time-varying covariates: LOCF carries forward dose-record values" begin
    using DataFrames

    # WT measured on dose record at t=0; not repeated on obs records → carry forward
    df = DataFrame(
        ID   = [1,   1,   1],
        TIME = [0.0, 6.0, 24.0],
        AMT  = [100.0, missing, missing],
        DV   = [missing, 2.5, 1.0],
        EVID = [1, 0, 0],
        MDV  = [1, 0, 0],
        WT   = [75.0, missing, missing],  # only present at baseline
    )

    pop = read_data(df)
    s1  = pop[1]
    # WT varies across the dataset? Only one non-missing value (75.0) — constant.
    @test haskey(s1.covariates, :wt)
    @test s1.covariates[:wt] ≈ 75.0
    @test isempty(s1.tvcov)
end

@testset "Time-varying covariates: predictions use per-obs covariate values" begin
    using DataFrames, ForwardDiff

    # Build a subject with two observations and a WT that changes between them
    doses = [DoseEvent(0.0, 100.0; cmt=1)]
    obs_times = [4.0, 24.0]
    obs_vals  = [2.5, 1.0]
    obs_cmts  = [1, 1]
    covariates = Dict{Symbol, Float64}()           # no time-constant covariates
    tvcov = Dict{Symbol, Vector{Float64}}(:wt => [70.0, 55.0])   # WT drops at t=24

    subject = Subject(1, doses, obs_times, obs_vals, obs_cmts, covariates, tvcov)

    # Model: CL = TVCL * (WT/70)^0.75,  V = TVV (no WT effect on V)
    model_str = """
    model TVCovTest
      [parameters]
        theta TVCL(5.0, 0.01, 100.0)
        theta TVV(50.0, 1.0, 500.0)
        omega ETA_CL ~ 0.0
        sigma PROP_ERR ~ 0.01
      [individual_parameters]
        CL = TVCL * (WT / 70.0)^0.75
        V  = TVV
      [structural_model]
        pk one_cpt_iv_bolus(cl=CL, v=V)
      [error_model]
        DV ~ proportional(PROP_ERR)
    end
    """
    model = parse_model_string(model_str)

    theta = [5.0, 50.0]
    eta   = [0.0]

    preds = JuliaNLME.compute_predictions(model, subject, theta, eta)

    # Expected: at each time, use the WT at that time to compute CL
    cl_t1 = 5.0 * (70.0/70.0)^0.75   # WT=70 at t=4
    cl_t2 = 5.0 * (55.0/70.0)^0.75   # WT=55 at t=24
    v     = 50.0
    dose  = 100.0
    expected_t1 = (dose/v) * exp(-(cl_t1/v) * 4.0)
    expected_t2 = (dose/v) * exp(-(cl_t2/v) * 24.0)

    @test preds[1] ≈ expected_t1 rtol=1e-8
    @test preds[2] ≈ expected_t2 rtol=1e-8

    # Verify predictions differ from what a constant-covariate model would give
    subject_const = Subject(1, doses, obs_times, obs_vals, obs_cmts,
                            Dict{Symbol,Float64}(:wt => 70.0))
    preds_const = JuliaNLME.compute_predictions(model, subject_const, theta, eta)
    @test preds[1] ≈ preds_const[1] rtol=1e-8                     # same WT at t=4
    @test !isapprox(preds[2], preds_const[2]; rtol=1e-3)           # different WT at t=24

    # AD compatibility: gradient through TV covariate path must be finite
    g = ForwardDiff.gradient(p -> sum(JuliaNLME.compute_predictions(model, subject, p, eta)),
                             theta)
    @test all(isfinite, g)
end

@testset "Time-varying covariates: full FOCE estimation smoke test" begin
    using DataFrames, Random
    Random.seed!(789)

    # Simulate 10 subjects with declining CRCL (kidney function worsening over time)
    # CL = TVCL * (CRCL/100)^0.5 * exp(ETA_CL)
    true_tvcl  = 4.0
    true_theta_crcl = 0.5
    true_omega = [0.10]
    true_sigma = [0.02]

    obs_times = [1.0, 4.0, 8.0, 24.0, 48.0]

    all_rows = []
    for id in 1:10
        crcl_baseline = 80.0 + 20.0 * randn()
        crcl_late     = max(crcl_baseline - 15.0 * rand(), 20.0)  # worsens by t=48
        eta_cl = sqrt(true_omega[1]) * randn()

        for (j, t) in enumerate(obs_times)
            # CRCL interpolated linearly between baseline and late
            crcl_t = t < 24 ? crcl_baseline : crcl_late
            cl = true_tvcl * (crcl_t / 100)^true_theta_crcl * exp(eta_cl)
            v  = 30.0
            dose = 100.0
            ipred = (dose/v) * exp(-(cl/v) * t)
            dv = ipred * (1 + sqrt(true_sigma[1]) * randn())
            push!(all_rows, (ID=id, TIME=t, AMT=missing, DV=max(dv, 0.001),
                             EVID=0, MDV=0, CMT=1, RATE=0.0, CRCL=crcl_t))
        end
        # Add dose record at t=0 (CRCL at baseline)
        pushfirst!(all_rows, (ID=id, TIME=0.0, AMT=100.0, DV=missing,
                               EVID=1, MDV=1, CMT=1, RATE=0.0, CRCL=crcl_baseline))
    end

    # Sort by ID and TIME
    df = sort(DataFrame(all_rows), [:ID, :TIME])

    pop = read_data(df)

    # CRCL should be detected as time-varying (it changes within subjects)
    @test haskey(pop[1].tvcov, :crcl)
    @test length(pop[1].tvcov[:crcl]) == length(obs_times)

    model_str = """
    model TVCRCLModel
      [parameters]
        theta TVCL(3.0, 0.01, 50.0)
        theta TVV(25.0, 1.0, 200.0)
        theta THETA_CRCL(0.4, 0.0, 2.0)
        omega ETA_CL ~ 0.15
        sigma PROP_ERR ~ 0.03
      [individual_parameters]
        CL = TVCL * (CRCL / 100.0)^THETA_CRCL * exp(ETA_CL)
        V  = TVV
      [structural_model]
        pk one_cpt_iv_bolus(cl=CL, v=V)
      [error_model]
        DV ~ proportional(PROP_ERR)
    end
    """
    model = parse_model_string(model_str)

    omega_init  = OmegaMatrix([0.15], [:ETA_CL]; diagonal=true)
    init_params = ModelParameters(
        [3.0, 25.0, 0.4], [:TVCL, :TVV, :THETA_CRCL],
        omega_init, SigmaMatrix([0.03], [:PROP_ERR])
    )

    result = fit(model, pop, init_params;
                 outer_maxiter=200, run_covariance_step=false, verbose=false)

    @test isfinite(result.ofv)
    @test all(isfinite, result.theta)
    # TVCL and THETA_CRCL should be in a reasonable ballpark
    @test 1.0 < result.theta[1] < 15.0   # TVCL
    @test 0.0 < result.theta[3] < 2.0    # THETA_CRCL
end

println("\nAll tests passed!")
