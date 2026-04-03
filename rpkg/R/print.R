#' Print a jnlme_fit result
#' @export
print.jnlme_fit <- function(x, digits = 4, ...) {
  status <- if (x$converged) "converged" else "NOT CONVERGED"
  cat(sprintf("JuliaNLME fit [%s]\n", status))
  cat(sprintf("  Model:      %s\n", x$model$name))
  cat(sprintf("  Method:     %s\n", if (x$interaction) "FOCE-I" else "FOCE"))
  cat(sprintf("  OFV: %-12s  AIC: %-12s  BIC: %s\n",
              round(x$ofv, digits), round(x$aic, digits), round(x$bic, digits)))
  cat(sprintf("  Subjects: %d   Observations: %d   Parameters: %d   Iterations: %d\n",
              x$n_subjects, x$n_obs, x$n_parameters, x$n_iterations))

  cat("\nTHETA:\n")
  if (!is.null(x$se_theta) && length(x$se_theta) > 0) {
    rse <- abs(x$se_theta / x$theta) * 100
    df  <- data.frame(
      Estimate = round(x$theta,    digits),
      SE       = round(x$se_theta, digits),
      `RSE%`   = round(rse, 1),
      check.names = FALSE
    )
    print(df)
  } else {
    print(round(x$theta, digits))
  }

  cat("\nOMEGA (variance-covariance):\n")
  print(round(x$omega, digits))

  cat("\nSIGMA:\n")
  if (!is.null(x$se_sigma) && length(x$se_sigma) > 0) {
    df <- data.frame(
      Estimate = round(x$sigma,    digits),
      SE       = round(x$se_sigma, digits),
      `RSE%`   = round(abs(x$se_sigma / x$sigma) * 100, 1),
      check.names = FALSE
    )
    print(df)
  } else {
    print(round(x$sigma, digits))
  }

  if (length(x$warnings) > 0) {
    cat("\nWarnings:\n")
    for (w in x$warnings) cat("  !", w, "\n")
  }

  invisible(x)
}

#' Summary of a jnlme_fit result
#' @export
summary.jnlme_fit <- function(object, ...) {
  print(object, ...)
}

#' Extract fixed-effect parameter estimates
#' @export
coef.jnlme_fit <- function(object, ...) {
  object$theta
}

#' Extract parameter covariance matrix
#'
#' Returns the covariance matrix of all packed parameters (THETA, Chol(OMEGA),
#' log(SIGMA)). Returns `NULL` if the covariance step failed.
#'
#' @export
vcov.jnlme_fit <- function(object, ...) {
  if (is.null(object$covariance_matrix)) NULL
  else object$covariance_matrix
}
