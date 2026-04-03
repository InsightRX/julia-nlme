#' Build a precompiled Julia sysimage for fast startup
#'
#' Creates a sysimage using PackageCompiler.jl that bakes JuliaNLME and its
#' dependencies into a single native library. Loading this sysimage on
#' subsequent sessions reduces startup from ~30-90 s to ~1-2 s.
#'
#' The function installs PackageCompiler.jl if needed, then calls
#' \code{create_sysimage()} with a warmup script that exercises the main
#' estimation and simulation code paths so that JIT-compiled code is included
#' in the image.
#'
#' This is a one-time operation that takes 5-15 minutes. Run it once after
#' installing or updating JuliaNLME, then point \code{jnlme_setup()} to the
#' generated file via the \code{sysimage_path} argument.
#'
#' @param path File path for the generated sysimage. Defaults to
#'   \code{"julianlme.so"} in the current working directory (use \code{".dll"}
#'   on Windows).
#' @param project Path to the JuliaNLME Julia project directory. Required if
#'   \code{jnlme_setup()} has not been called yet in this session.
#' @param julia_home Path to the Julia executable directory. Defaults to Julia
#'   on PATH.
#' @param verbose Print PackageCompiler progress. Default \code{TRUE}.
#'
#' @return The path to the created sysimage, invisibly. Pass this to
#'   \code{jnlme_setup(sysimage_path = ...)} in future sessions.
#'
#' @examples
#' \dontrun{
#' # Build once (takes 5-15 minutes)
#' sysimage <- jnlme_build_sysimage(
#'   path    = "julianlme.so",
#'   project = "/path/to/julia-nlme"
#' )
#'
#' # Use in all future sessions
#' jnlme_setup(project = "/path/to/julia-nlme", sysimage_path = sysimage)
#' }
#'
#' @export
jnlme_build_sysimage <- function(path    = file.path(getwd(), .default_sysimage_name()),
                                   project = Sys.getenv("JULIA_PROJECT", unset = NA),
                                   julia_home = NULL,
                                   verbose = TRUE) {

  path <- normalizePath(path, mustWork = FALSE)

  # Ensure Julia is running (without a sysimage — we're about to build one)
  if (!.jnlme$initialized) {
    message("julianlme: initializing Julia to build sysimage...")
    jnlme_setup(project = project, julia_home = julia_home)
  }

  # Install PackageCompiler if not present
  have_pc <- julia_eval(
    'try; using PackageCompiler; true; catch; false; end'
  )
  if (!have_pc) {
    message("julianlme: installing PackageCompiler.jl...")
    julia_eval('import Pkg; Pkg.add("PackageCompiler")', need_return = "None")
    julia_eval('using PackageCompiler', need_return = "None")
  }

  warmup_path <- system.file("julia/precompile_warmup.jl", package = "julianlme")
  warmup_path <- gsub("\\\\", "/", warmup_path)
  out_path    <- gsub("\\\\", "/", path)

  message(sprintf(
    "julianlme: building sysimage at '%s'.\n  This takes 5-15 minutes — please wait...",
    path
  ))

  julia_eval(sprintf(
    'using PackageCompiler; create_sysimage(
      ["JuliaNLME", "DataFrames", "CSV"],
      sysimage_path              = "%s",
      precompile_execution_file  = "%s"
    )',
    out_path, warmup_path
  ), need_return = "None")

  if (!file.exists(path)) {
    stop("julianlme: sysimage was not created — check Julia output for errors.",
         call. = FALSE)
  }

  message(sprintf(
    "julianlme: sysimage built successfully.\n",
    "  Use it in future sessions with:\n",
    "  jnlme_setup(project = ..., sysimage_path = \"%s\")", path
  ))

  invisible(path)
}

# Platform-appropriate default filename
.default_sysimage_name <- function() {
  if (.Platform$OS.type == "windows") "julianlme.dll" else "julianlme.so"
}
