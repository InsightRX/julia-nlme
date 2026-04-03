#' Simulate observations from a fitted model
#'
#' Draws individual parameters from the fitted population distribution and
#' simulates observations for the provided dataset.
#'
#' @param model A `jnlme_model` object.
#' @param data A data.frame in NONMEM format defining the dosing and observation
#'   schedule. DV values in observation rows are replaced by simulated values.
#' @param result A `jnlme_fit` object. Population parameters (THETA, OMEGA,
#'   SIGMA) are taken from the fitted estimates.
#' @param n_sims Number of simulation replicates. Each replicate draws a new
#'   set of individual parameters from the population distribution. Default `1`.
#'
#' @return A data.frame with one row per observation per replicate. Includes
#'   all original observation-row columns plus `pred` (population prediction),
#'   `ipred` (individual prediction), `dv` (simulated observation), and `_sim`
#'   (replicate index, 1 to `n_sims`).
#'
#' @examples
#' \dontrun{
#' model  <- jnlme_model("warfarin_oral.jnlme")
#' result <- jnlme_fit(model, data)
#'
#' # Single simulation
#' sim <- jnlme_simulate(model, data, result)
#'
#' # 100 replicates (for VPC)
#' vpc_sims <- jnlme_simulate(model, data, result, n_sims = 100)
#' }
#'
#' @export
jnlme_simulate <- function(model, data, result, n_sims = 1L) {
  .check_setup()
  stopifnot(
    inherits(model,  "jnlme_model"),
    inherits(result, "jnlme_fit"),
    is.data.frame(data)
  )

  csv <- .df_to_csv(data)
  on.exit(unlink(csv))

  out_path <- julia_call("r_simulate", model$key, result$.handle, csv,
                          n_sims = as.integer(n_sims))
  sim_df <- utils::read.csv(out_path, stringsAsFactors = FALSE)
  unlink(out_path)
  sim_df
}
