"""
Plotting utilities for JuliaNLME diagnostics.

Requires TidierPlots (already a package dependency).
"""

using TidierPlots

# ---------------------------------------------------------------------------
# VPC plot
# ---------------------------------------------------------------------------

"""
    plot_vpc(vpc_result; kwargs...) → ggplot object

Plot a Visual Predictive Check from a [`VPCResult`](@ref).

The plot shows:
- **Shaded ribbons**: 90% confidence interval around the simulated 5th, 50th,
  and 95th percentiles (or whichever `pi` was used when calling `vpc()`).
- **Solid/dashed blue lines**: simulated median of each percentile band.
- **Red lines**: observed percentiles (dashed for PI bounds, solid for median).
- **Grey points**: raw observed data (disable with `obs_points=false`).

When the `VPCResult` was computed with `stratify`, each stratum is shown in
its own facet panel.

## Keyword arguments

| Keyword | Default | Description |
|---------|---------|-------------|
| `obs_points` | `true` | Overlay raw observed data as scatter points |
| `log_y` | `false` | Log₁₀ scale on the y-axis |
| `x_lab` | `"Time"` | x-axis label |
| `y_lab` | `"Concentration"` | y-axis label |
| `title` | `"Visual Predictive Check"` | Plot title |
| `pi_fill` | `"#5B9BD5"` | Fill colour for the PI ribbon bands |
| `pi_line_color` | `"#1A6099"` | Line colour for simulated percentile lines |
| `obs_line_color` | `"#C0392B"` | Line colour for observed percentile lines |
| `obs_point_color` | `"#555555"` | Colour for observed scatter points |
| `sim_median` | `false` | Show median lines of the simulated CI bands |
| `ribbon_alpha` | `0.25` | Transparency of PI ribbons |
| `point_alpha` | `0.35` | Transparency of observed scatter points |
| `point_size` | `1.5` | Size of observed scatter points |
"""
function plot_vpc(v::VPCResult;
                  obs_points::Bool      = true,
                  sim_median::Bool      = false,
                  log_y::Bool           = false,
                  x_lab::String         = "Time",
                  y_lab::String         = "Concentration",
                  title::String         = "Visual Predictive Check",
                  pi_fill::String       = "#5B9BD5",
                  pi_line_color::String = "#1A6099",
                  obs_line_color::String = "#C0392B",
                  obs_point_color::String = "#555555",
                  ribbon_alpha::Float64 = 0.25,
                  point_alpha::Float64  = 0.35,
                  point_size::Float64   = 1.5)

    stat   = v.vpc_stat
    ostat  = v.obs_stat
    obs    = v.obs
    is_stratified = :strat in propertynames(stat)

    isempty(stat)  && error("vpc_result has no simulation data — cannot plot")
    isempty(ostat) && error("vpc_result has no observed data — cannot plot")

    p = ggplot()

    # ---- Simulated PI confidence interval ribbons ----
    # Lower PI band (pi_lo_lo → pi_lo_hi)
    p = p + geom_ribbon(stat,
                @aes(x = bin_mid, ymin = pi_lo_lo, ymax = pi_lo_hi),
                fill = pi_fill, alpha = ribbon_alpha)

    # Upper PI band (pi_hi_lo → pi_hi_hi)
    p = p + geom_ribbon(stat,
                @aes(x = bin_mid, ymin = pi_hi_lo, ymax = pi_hi_hi),
                fill = pi_fill, alpha = ribbon_alpha)

    # Median band (p50_lo → p50_hi)
    p = p + geom_ribbon(stat,
                @aes(x = bin_mid, ymin = p50_lo, ymax = p50_hi),
                fill = pi_fill, alpha = ribbon_alpha)

    # ---- Simulated percentile median lines (optional) ----
    if sim_median
        p = p + geom_line(stat,
                    @aes(x = bin_mid, y = pi_lo_med),
                    color = pi_line_color, linetype = :dash)
        p = p + geom_line(stat,
                    @aes(x = bin_mid, y = p50_med),
                    color = pi_line_color)
        p = p + geom_line(stat,
                    @aes(x = bin_mid, y = pi_hi_med),
                    color = pi_line_color, linetype = :dash)
    end

    # ---- Observed percentile lines ----
    p = p + geom_line(ostat,
                @aes(x = bin_mid, y = obs_pi_lo),
                color = obs_line_color, linetype = :dash)
    p = p + geom_line(ostat,
                @aes(x = bin_mid, y = obs_p50),
                color = obs_line_color)
    p = p + geom_line(ostat,
                @aes(x = bin_mid, y = obs_pi_hi),
                color = obs_line_color, linetype = :dash)

    # ---- Raw observed scatter ----
    if obs_points
        p = p + geom_point(obs,
                    @aes(x = time, y = dv),
                    color = obs_point_color,
                    alpha = point_alpha,
                    size  = point_size)
    end

    # ---- Scales, labels, facets ----
    if log_y
        p = p + scale_y_log10()
    end

    p = p + labs(x = x_lab, y = y_lab, title = title)

    if is_stratified
        p = p + facet_wrap(:strat)
    end

    return p
end
