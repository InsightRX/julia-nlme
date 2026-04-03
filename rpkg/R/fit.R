# Internal: write a data.frame to a temp CSV and return its path
.df_to_csv <- function(data) {
  path <- tempfile(fileext = ".csv")
  utils::write.csv(data, path, row.names = FALSE, na = ".")
  path
}

# Internal: build a jnlme_fit object from the raw list returned by Julia
.build_fit <- function(raw, model) {
  n_eta  <- length(model$eta_names)
  n_om   <- raw$omega_dim

  omega <- matrix(raw$omega, nrow = n_om, ncol = n_om,
                  dimnames = list(model$eta_names, model$eta_names))

  theta  <- stats::setNames(raw$theta,    raw$theta_names)
  sigma  <- stats::setNames(raw$sigma,    raw$sigma_names)
  se_th  <- if (length(raw$se_theta) > 0) stats::setNames(raw$se_theta,  raw$theta_names) else NULL
  se_sig <- if (length(raw$se_sigma) > 0) stats::setNames(raw$se_sigma,  raw$sigma_names) else NULL
  se_om  <- raw$se_omega  # Cholesky elements — returned as-is

  diagnostics <- utils::read.csv(raw$diagnostics_csv, stringsAsFactors = FALSE)
  etas        <- utils::read.csv(raw$eta_csv,          stringsAsFactors = FALSE)

  structure(
    list(
      model        = model,
      converged    = raw$converged,
      ofv          = raw$ofv,
      aic          = raw$aic,
      bic          = raw$bic,
      theta        = theta,
      omega        = omega,
      sigma        = sigma,
      se_theta     = se_th,
      se_omega     = se_om,
      se_sigma     = se_sig,
      n_obs        = raw$n_obs,
      n_subjects   = raw$n_subjects,
      n_parameters = raw$n_parameters,
      n_iterations = raw$n_iterations,
      interaction  = raw$interaction,
      warnings     = raw$warnings,
      diagnostics  = diagnostics,   # long-format: id, ipred, pred, iwres, cwres
      etas         = etas,          # wide-format: id, eta_<name>, ...
      .handle      = raw[["_handle"]]
    ),
    class = "jnlme_fit"
  )
}

#' Fit a model using FOCE or FOCE-I
#'
#' @param model A `jnlme_model` object from `jnlme_model()`.
#' @param data A data.frame in NONMEM format (ID, TIME, DV, EVID, AMT, ...).
#' @param interaction Logical. Use FOCE-I (eta-epsilon interaction). Recommended
#'   for proportional and combined error models. Default `FALSE`.
#' @param outer_maxiter Maximum outer BFGS iterations. Default `500`.
#' @param outer_gtol Gradient tolerance for the outer optimizer. Default `1e-6`.
#' @param inner_maxiter Maximum EBE iterations per subject. Default `200`.
#' @param run_covariance_step Compute standard errors via Hessian inversion.
#'   Default `TRUE`.
#' @param nthreads Number of Julia threads for per-subject parallelism. Start
#'   Julia with multiple threads via `jnlme_setup(nthreads = N)`. Default `1`.
#' @param verbose Print iteration progress. Default `TRUE`.
#'
#' @return An object of class `jnlme_fit`.
#'
#' @examples
#' \dontrun{
#' model  <- jnlme_model("warfarin_oral.jnlme")
#' result <- jnlme_fit(model, data, interaction = TRUE)
#' print(result)
#' }
#'
#' @export
jnlme_fit <- function(model, data,
                       interaction          = FALSE,
                       outer_maxiter        = 500L,
                       outer_gtol           = 1e-6,
                       inner_maxiter        = 200L,
                       run_covariance_step  = TRUE,
                       nthreads             = 1L,
                       verbose              = TRUE) {
  .check_setup()
  stopifnot(inherits(model, "jnlme_model"), is.data.frame(data))

  csv <- .df_to_csv(data)
  on.exit(unlink(csv))

  raw <- julia_call("r_fit", model$key, csv,
                    interaction         = interaction,
                    outer_maxiter       = as.integer(outer_maxiter),
                    outer_gtol          = as.double(outer_gtol),
                    inner_maxiter       = as.integer(inner_maxiter),
                    run_covariance_step = run_covariance_step,
                    nthreads            = as.integer(nthreads),
                    verbose             = verbose)
  .build_fit(raw, model)
}

#' Fit a model using ITS (Iterative Two-Stage)
#'
#' ITS is a fast, deterministic EM algorithm. Useful as a quick first estimate
#' or as a warm-start for FOCE.
#'
#' @inheritParams jnlme_fit
#' @param n_iter Maximum ITS iterations. Default `100`.
#' @param conv_window Rolling window length for convergence check. Default `20`.
#' @param rel_tol Relative parameter change tolerance. Default `1e-4`.
#'
#' @return An object of class `jnlme_fit`.
#'
#' @export
jnlme_fit_its <- function(model, data,
                            n_iter               = 100L,
                            conv_window          = 20L,
                            rel_tol              = 1e-4,
                            run_covariance_step  = TRUE,
                            interaction          = FALSE,
                            nthreads             = 1L,
                            verbose              = TRUE) {
  .check_setup()
  stopifnot(inherits(model, "jnlme_model"), is.data.frame(data))

  csv <- .df_to_csv(data)
  on.exit(unlink(csv))

  raw <- julia_call("r_fit_its", model$key, csv,
                    n_iter              = as.integer(n_iter),
                    conv_window         = as.integer(conv_window),
                    rel_tol             = as.double(rel_tol),
                    run_covariance_step = run_covariance_step,
                    interaction         = interaction,
                    nthreads            = as.integer(nthreads),
                    verbose             = verbose)
  .build_fit(raw, model)
}

#' Fit a model using SAEM (Stochastic Approximation EM)
#'
#' SAEM is a stochastic EM algorithm that can be more robust than FOCE for
#' models with complex random-effects structures.
#'
#' @inheritParams jnlme_fit
#' @param n_iter_exploration Phase-1 iterations (step size = 1). Default `150`.
#' @param n_iter_convergence Phase-2 iterations (decreasing step size).
#'   Default `250`.
#' @param n_mh_steps Metropolis-Hastings steps per subject per iteration.
#'   Default `2`.
#'
#' @return An object of class `jnlme_fit`.
#'
#' @export
jnlme_fit_saem <- function(model, data,
                             n_iter_exploration   = 150L,
                             n_iter_convergence   = 250L,
                             n_mh_steps           = 2L,
                             run_covariance_step  = TRUE,
                             interaction          = FALSE,
                             nthreads             = 1L,
                             verbose              = TRUE) {
  .check_setup()
  stopifnot(inherits(model, "jnlme_model"), is.data.frame(data))

  csv <- .df_to_csv(data)
  on.exit(unlink(csv))

  raw <- julia_call("r_fit_saem", model$key, csv,
                    n_iter_exploration  = as.integer(n_iter_exploration),
                    n_iter_convergence  = as.integer(n_iter_convergence),
                    n_mh_steps          = as.integer(n_mh_steps),
                    run_covariance_step = run_covariance_step,
                    interaction         = interaction,
                    nthreads            = as.integer(nthreads),
                    verbose             = verbose)
  .build_fit(raw, model)
}
