"""
Build a precompiled system image for fast jnlme startup.

Usage (from the package root):
  julia --project=. scripts/build_sysimage.jl

The resulting sysimage is saved to build/jnlme.so.
The bin/jnlme shell script detects and uses it automatically.

Startup time comparison:
  Without sysimage: ~8–15 s (full Julia + package compilation)
  With sysimage:    ~0.5–2 s (all packages pre-compiled)

Rebuilding: re-run this script after adding dependencies or making
major structural changes to the package.
"""

import Pkg
Pkg.add("PackageCompiler")
using PackageCompiler

root = dirname(@__DIR__)
out  = joinpath(root, "build", "jnlme.so")
mkpath(joinpath(root, "build"))

# Precompile script: a minimal fit run that exercises the full call stack.
precompile_script = joinpath(root, "scripts", "_precompile_workload.jl")

@info "Building sysimage → $out  (this takes 5–15 minutes the first time)"
create_sysimage(
    [:JuliaNLME, :ArgParse];
    sysimage_path              = out,
    precompile_execution_file  = precompile_script,
    project                    = root,
)
@info "Done. System image saved to $out"
@info "Startup test: time bin/jnlme --help"
