#' Simulate observations from a model
#'
#' Draws individual parameters from the population distribution and simulates
#' observations for the provided dataset. Accepts either a fitted result or
#' explicit parameter values.
#'
#' @param model A `jnlme_model` object.
#' @param data A data.frame in NONMEM format defining the dosing and observation
#'   schedule. DV values in observation rows are replaced by simulated values.
#' @param params A `jnlme_fit` object **or** a named list with elements:
#'   \describe{
#'     \item{`theta`}{Named numeric vector of fixed-effect values.}
#'     \item{`omega`}{Diagonal BSV variances as a named numeric vector (one
#'       value per ETA), or a full named variance-covariance matrix.}
#'     \item{`sigma`}{Named numeric vector of residual variance values.}
#'   }
#' @param n_sims Number of simulation replicates. Default `1`.
#'
#' @return A data.frame with one row per observation per replicate, including
#'   `pred`, `ipred`, `dv` (simulated), and `_sim` (replicate index).
#'
#' @examples
#' \dontrun{
#' model <- jnlme_model("warfarin_oral.jnlme")
#'
#' # Simulate from known true parameter values
#' sim <- jnlme_simulate(model, data, list(
#'   theta = c(TVCL = 0.134, TVV = 8.1, TVKA = 1.0),
#'   omega = c(ETA_CL = 0.07, ETA_V = 0.02, ETA_KA = 0.40),
#'   sigma = c(PROP_ERR = 0.01)
#' ))
#'
#' # Simulate from a fitted result (100 replicates for VPC)
#' result  <- jnlme_fit(model, data)
#' vpc_sims <- jnlme_simulate(model, data, result, n_sims = 100)
#' }
#'
#' @export
jnlme_simulate <- function(model, data, params, n_sims = 1L) {
  .check_setup()
  stopifnot(inherits(model, "jnlme_model"), is.data.frame(data))

  csv <- .df_to_csv(data)
  on.exit(unlink(csv))

  if (inherits(params, "jnlme_fit")) {
    out_path <- julia_call("r_simulate", model$key, params$.handle, csv,
                            n_sims = as.integer(n_sims))
  } else if (is.list(params)) {
    if (!all(c("theta", "omega", "sigma") %in% names(params)))
      stop("params list must have elements 'theta', 'omega', and 'sigma'.", call. = FALSE)
    # omega may be a full matrix (take diagonal) or already a vector of variances
    omega_diag <- if (is.matrix(params$omega)) diag(params$omega) else params$omega
    out_path <- julia_call("r_simulate_params",
                            model$key, csv,
                            as.double(params$theta),
                            as.double(omega_diag),
                            as.double(params$sigma),
                            n_sims = as.integer(n_sims))
  } else {
    stop("'params' must be a jnlme_fit object or a list with theta, omega, sigma.",
         call. = FALSE)
  }

  sim_df <- utils::read.csv(out_path, stringsAsFactors = FALSE)
  unlink(out_path)
  sim_df
}
