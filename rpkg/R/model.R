#' Parse a JuliaNLME model file
#'
#' Reads a `.jnlme` model file, compiles it in Julia, and returns an R object
#' that can be passed to `jnlme_fit()` and `jnlme_simulate()`.
#'
#' @param path Path to a `.jnlme` model file.
#'
#' @return An object of class `jnlme_model`.
#'
#' @examples
#' \dontrun{
#' model <- jnlme_model("warfarin_oral.jnlme")
#' print(model)
#' }
#'
#' @export
jnlme_model <- function(path) {
  .check_setup()
  path <- normalizePath(path, mustWork = TRUE)
  key  <- julia_call("r_parse_model", path)
  info <- julia_call("r_model_info", key)

  structure(
    list(
      key          = key,
      path         = path,
      name         = info$name,
      pk_model     = info$pk_model,
      error_model  = info$error_model,
      theta_names  = info$theta_names,
      eta_names    = info$eta_names,
      sigma_names  = info$sigma_names,
      theta_init   = stats::setNames(info$theta_init, info$theta_names),
      theta_lower  = stats::setNames(info$theta_lower, info$theta_names),
      theta_upper  = stats::setNames(info$theta_upper, info$theta_names),
      omega_init   = matrix(info$omega_init,
                            nrow = info$n_eta, ncol = info$n_eta,
                            dimnames = list(info$eta_names, info$eta_names)),
      sigma_init   = stats::setNames(info$sigma_init, info$sigma_names)
    ),
    class = "jnlme_model"
  )
}

#' @export
print.jnlme_model <- function(x, ...) {
  cat("JuliaNLME model:", x$name, "\n")
  cat("  PK model:    ", x$pk_model,    "\n")
  cat("  Error model: ", x$error_model, "\n")
  cat("  THETA:       ", paste(x$theta_names, collapse = ", "), "\n")
  cat("  ETA:         ", paste(x$eta_names,   collapse = ", "), "\n")
  cat("  SIGMA:       ", paste(x$sigma_names, collapse = ", "), "\n")
  invisible(x)
}
