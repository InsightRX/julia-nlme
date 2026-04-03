using Documenter
using JuliaNLME

makedocs(;
    sitename = "JuliaNLME.jl",
    modules  = [JuliaNLME],
    authors  = "InsightRX",
    format   = Documenter.HTML(;
        prettyurls = true,
        canonical  = "https://insightrx.github.io/julia-nlme",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Interfaces" => "interfaces.md",
        "Dataset Format" => "dataset_format.md",
        "Model File Format" => "model_file_format.md",
        "Estimation Methods" => "estimation_methods.md",
        "Diagnostics" => "diagnostics.md",
        "Examples" => [
            "1-Cpt Oral (Warfarin)" => "examples/warfarin.md",
            "2-Cpt IV Bolus" => "examples/two_cpt_iv.md",
            "2-Cpt Oral + Covariates" => "examples/two_cpt_oral_cov.md",
            "ODE: Michaelis-Menten" => "examples/ode_mm.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)
