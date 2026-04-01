"""
Output formatting: parameter tables, subject-level results tables.
"""

using DataFrames, Printf

# ---------------------------------------------------------------------------
# Parameter estimates table (NONMEM-style)
# ---------------------------------------------------------------------------

"""
    parameter_table(result)

Return a DataFrame with fixed-effect estimates and standard errors.
"""
function parameter_table(result::FitResult)::DataFrame
    has_se = !isempty(result.se_theta)

    rows = []
    n_theta = length(result.theta)

    # THETA
    for i in 1:n_theta
        name = string(result.theta_names[i])
        est  = result.theta[i]
        se   = has_se ? result.se_theta[i] : NaN
        rse  = has_se && est != 0 ? abs(se / est) * 100 : NaN
        push!(rows, (Parameter=name, Estimate=est, SE=se, RSE_pct=rse, Type="THETA"))
    end

    # OMEGA (lower triangle, or diagonal-only when omega is diagonal)
    n_eta = size(result.omega, 1)
    is_diag_omega = has_se && length(result.se_omega) == n_eta
    se_idx = 1
    for i in 1:n_eta, j in 1:i
        is_diag_elem = (i == j)
        # Skip off-diagonal rows when using diagonal omega (no estimated covariances)
        is_diag_omega && !is_diag_elem && continue
        name = "OMEGA($i,$j)"
        est  = result.omega[i, j]
        se   = has_se && se_idx <= length(result.se_omega) ? result.se_omega[se_idx] : NaN
        rse  = has_se && est != 0 ? abs(se / est) * 100 : NaN
        push!(rows, (Parameter=name, Estimate=est, SE=se, RSE_pct=rse, Type="OMEGA"))
        se_idx += 1
    end

    # SIGMA
    for (i, est) in enumerate(result.sigma)
        name = i <= length(result.model.eta_names) ? "SIGMA($i)" : "SIGMA($i)"
        se   = has_se && i <= length(result.se_sigma) ? result.se_sigma[i] : NaN
        rse  = has_se && est != 0 ? abs(se / est) * 100 : NaN
        push!(rows, (Parameter=name, Estimate=est, SE=se, RSE_pct=rse, Type="SIGMA"))
    end

    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Subject-level results table (sdtab equivalent)
# ---------------------------------------------------------------------------

"""
    sdtab(result, population)

Return a long-format DataFrame with one row per observation:
  ID, TIME, DV, PRED, IPRED, CWRES, IWRES, ETA_*
"""
function sdtab(result::FitResult, population::Population)::DataFrame
    rows = []
    n_eta = size(result.omega, 1)

    for (i, (subject, sub_res)) in enumerate(zip(population.subjects, result.subjects))
        for j in eachindex(subject.observations)
            row = Dict{String, Any}(
                "ID"    => subject.id,
                "TIME"  => subject.obs_times[j],
                "DV"    => subject.observations[j],
                "PRED"  => sub_res.pred[j],
                "IPRED" => sub_res.ipred[j],
                "CWRES" => sub_res.cwres[j],
                "IWRES" => sub_res.iwres[j],
            )
            for k in 1:n_eta
                row["ETA$(k)"] = sub_res.eta[k]
            end
            push!(rows, row)
        end
    end

    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

"""
    print_results(result)

Print a NONMEM-style parameter estimate summary to stdout.
"""
function print_results(result::FitResult)
    println("\n" * "="^60)
    println("  FOCE Estimation Results: $(result.model.name)")
    println("="^60)

    method = result.interaction ? "FOCE-I (eta-epsilon interaction)" : "FOCE"
    status = result.converged ? "SUCCESSFUL" : "FAILED"
    @printf "  Method:       %s\n" method
    @printf "  Minimisation: %s\n" status
    @printf "  OFV (-2LL):   %.3f\n"  result.ofv
    @printf "  AIC:          %.3f\n"  result.aic
    @printf "  BIC:          %.3f\n"  result.bic
    @printf "  Subjects:     %d\n"    result.n_subjects
    @printf "  Observations: %d\n"    result.n_obs
    @printf "  Parameters:   %d\n"    result.n_parameters
    @printf "  Iterations:   %d\n"    result.n_iterations

    has_se = !isempty(result.se_theta)
    println("\n  THETA:")
    println(has_se ? "    Name              Estimate        SE          %RSE" :
                     "    Name              Estimate")
    println("  " * "-"^(has_se ? 55 : 30))
    for i in 1:length(result.theta)
        name = string(result.theta_names[i])
        est  = result.theta[i]
        if has_se
            se  = result.se_theta[i]
            rse = est != 0 ? abs(se/est)*100 : NaN
            @printf "    %-16s  %12.4g  %10.4g  %8.1f%%\n" name est se rse
        else
            @printf "    %-16s  %12.4g\n" name est
        end
    end

    println("\n  OMEGA (variance):")
    n_eta = size(result.omega, 1)
    for i in 1:n_eta
        @printf "    OMEGA(%d,%d) = %.4g\n" i i result.omega[i,i]
    end

    println("\n  SIGMA:")
    for (i, v) in enumerate(result.sigma)
        @printf "    SIGMA(%d) = %.4g\n" i v
    end

    if !isempty(result.warnings)
        println("\n  WARNINGS:")
        for w in result.warnings; println("    ! $w"); end
    end

    println("="^60 * "\n")
end
