# Internal environment for package state
.jnlme <- new.env(parent = emptyenv())
.jnlme$initialized <- FALSE

#' Initialize the Julia runtime and load JuliaNLME
#'
#' Must be called once per R session before any other julianlme function.
#' Julia must be installed and on PATH (or its location given via `julia_home`).
#'
#' @param project Path to the JuliaNLME Julia project directory (the folder
#'   containing `Project.toml`). Defaults to the JULIA_PROJECT environment
#'   variable if set, otherwise uses Julia's default environment.
#' @param julia_home Path to the directory containing the Julia executable.
#'   Defaults to the JULIA_HOME environment variable or Julia on PATH.
#' @param nthreads Number of Julia threads to start with. Passed as
#'   JULIA_NUM_THREADS. Defaults to 1; set higher for faster estimation
#'   (requires Julia to support multi-threading, which it does by default).
#' @param ... Additional arguments forwarded to `JuliaCall::julia_setup()`.
#'
#' @examples
#' \dontrun{
#' # Julia project path where JuliaNLME is installed
#' jnlme_setup(project = "/path/to/julia-nlme", nthreads = 4)
#' }
#'
#' @export
jnlme_setup <- function(project = Sys.getenv("JULIA_PROJECT", unset = NA),
                         julia_home = NULL,
                         nthreads = 1,
                         ...) {
  if (.jnlme$initialized) {
    message("julianlme: Julia already initialized.")
    return(invisible(NULL))
  }

  if (nthreads > 1) {
    old_threads <- Sys.getenv("JULIA_NUM_THREADS", unset = NA)
    Sys.setenv(JULIA_NUM_THREADS = nthreads)
    on.exit({
      if (is.na(old_threads)) Sys.unsetenv("JULIA_NUM_THREADS")
      else Sys.setenv(JULIA_NUM_THREADS = old_threads)
    })
  }

  julia_setup(JULIA_HOME = julia_home, ...)

  # Activate the Julia project if given
  if (!is.na(project) && nchar(project) > 0) {
    project <- normalizePath(project, mustWork = TRUE)
    julia_eval(sprintf('import Pkg; Pkg.activate("%s", io=devnull)',
                       gsub("\\\\", "/", project)))
  }

  julia_eval("using JuliaNLME", need_return = FALSE)

  # Load bridge functions
  bridge_path <- system.file("julia/bridge.jl", package = "julianlme")
  julia_source(bridge_path)

  .jnlme$initialized <- TRUE
  message("julianlme: Julia initialized with JuliaNLME.")
  invisible(NULL)
}

# Internal: assert setup has been called
.check_setup <- function() {
  if (!.jnlme$initialized) {
    stop("julianlme: call jnlme_setup() first.", call. = FALSE)
  }
}

#' Release cached Julia model and result objects
#'
#' Frees memory held by Julia-side caches (compiled models and fit results).
#' Call this between unrelated analyses to avoid accumulating stale objects.
#'
#' @export
jnlme_clear_cache <- function() {
  .check_setup()
  julia_call("r_clear_cache", need_return = FALSE)
  invisible(NULL)
}
