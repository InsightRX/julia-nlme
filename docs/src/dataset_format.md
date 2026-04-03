# Dataset Format

JuliaNLME uses a NONMEM-compatible tabular format. Each row represents either a dosing event or an observation for one subject. Datasets can be loaded from a CSV file or supplied as a Julia `DataFrame` — the column names are case-insensitive in both cases.

## Required Columns

| Column | Description |
|--------|-------------|
| `ID` | Subject identifier (integer) |
| `TIME` | Time of event or observation (any consistent unit) |
| `DV` | Dependent variable (observed concentration or response). Use `.`, `NA`, or leave empty for missing values. |

## Event Type: EVID

`EVID` controls how each row is interpreted:

| Value | Meaning |
|-------|---------|
| `0` | Observation record. `DV` is the measured value. |
| `1` | Dose record. `AMT` defines the dose; `RATE` and `CMT` are optional. |
| `4` | Reset + dose. Clears all compartment amounts to zero, then applies the dose. Useful for multiple-dose studies where the washout is complete. |
| `2` | Other event — row is read but ignored during estimation. |

If `EVID` is absent, all rows are treated as observation records (`EVID=0`).

## Dose Columns

These columns are only relevant on dose rows (`EVID=1` or `EVID=4`):

| Column | Default | Description |
|--------|---------|-------------|
| `AMT` | — | Dose amount (same units as the structural model) |
| `CMT` | `1` | Compartment number receiving the dose |
| `RATE` | `0` | Infusion rate. `0` = bolus (instantaneous). `RATE > 0` means constant-rate infusion; the duration is derived as `AMT / RATE`. |
| `II` | `0` | Inter-dose interval, used with `SS=1` for steady-state dosing |
| `SS` | `0` | `1` = apply steady-state pre-dosing before this event |

## Observation Columns

| Column | Default | Description |
|--------|---------|-------------|
| `MDV` | `0` | Missing dependent variable flag. `MDV=1` excludes the row from the likelihood (e.g., BLQ observations, time-zero samples). |

## Covariates

Any column that is not one of the reserved names above is treated as a covariate and passed to the structural/individual-parameter model. Column names are normalized to lowercase internally, but model files use uppercase names (e.g., `WT`, `CRCL`) — the parser handles the mapping automatically.

**Time-constant covariates** — value is the same across all rows for a subject (or only specified on dose rows). The first non-missing value per subject is used.

**Time-varying covariates** — value changes across observation rows within a subject. The last-observation-carried-forward (LOCF) rule is applied: at each observation time, the most recent non-missing covariate value is used.

JuliaNLME automatically detects which covariates are time-constant vs time-varying by checking whether values change within any subject.

## Minimal Example

A one-compartment oral model with a single dose and five observations:

```
ID,TIME,AMT,DV,EVID,MDV
1,0,100,.,1,1
1,0.5,.,0.52,0,0
1,1,.,0.89,0,0
1,2,.,1.10,0,0
1,4,.,0.88,0,0
1,8,.,0.51,0,0
2,0,100,.,1,1
2,0.5,.,0.41,0,0
...
```

Missing values in `DV` on dose rows can be `.`, `NA`, or empty — they are ignored.

## With Covariates

```
ID,TIME,AMT,DV,EVID,MDV,WT,CRCL
1,0,100,.,1,1,70,95
1,0.5,.,0.52,0,0,70,95
1,1,.,0.89,0,0,70,95
2,0,100,.,1,1,58,72
2,0.5,.,0.41,0,0,58,72
```

Covariates only need to be specified on dose rows if they are time-constant — the reader carries them forward automatically.

## Infusion Dosing

For constant-rate infusions, set `RATE > 0` on the dose row. The infusion duration is `AMT / RATE`:

```
ID,TIME,AMT,DV,EVID,MDV,CMT,RATE
1,0,500,.,1,1,1,50
1,1,.,12.3,0,0,,
1,4,.,8.7,0,0,,
```

Here a 500 mg dose is given at rate 50 mg/h (10-hour infusion). Observation rows can leave `RATE` and `CMT` blank.

## Multiple Doses

Add one `EVID=1` row per dose. Times must be non-decreasing within a subject:

```
ID,TIME,AMT,DV,EVID,MDV
1,0,100,.,1,1
1,12,.,0.34,0,0
1,24,100,.,1,1
1,36,.,0.28,0,0
```

## Steady-State Dosing

Use `SS=1` and `II` to apply a steady-state pre-dose condition without simulating the full dosing history:

```
ID,TIME,AMT,DV,EVID,MDV,II,SS
1,0,100,.,1,1,24,1
1,2,.,4.2,0,0,,
1,8,.,1.9,0,0,,
```

This applies the steady-state concentration for a 100 mg every-24-hour regimen before the first measured observation.

## BLQ and Missing Observations

Rows with `MDV=1` are excluded from the likelihood but remain in the dataset. Use this for:

- **BLQ samples**: keep the row (preserves the design) but exclude it from fitting
- **Dose-time observations**: dose rows with `DV` set to `.` typically have `MDV=1`

```
ID,TIME,AMT,DV,EVID,MDV
1,0,100,.,1,1
1,0.5,.,0.05,0,1
1,1,.,0.89,0,0
```

Row at `TIME=0.5` is present but excluded from the likelihood (`MDV=1`).

## Loading Data

```julia
using JuliaNLME

# From CSV file
result = fit(model, "data.csv")

# From a DataFrame (no explicit read_data call needed)
using DataFrames, CSV
df = CSV.read("data.csv", DataFrame)
result = fit(model, df)

# Specifying which columns are covariates (all others ignored)
result = fit(model, "data.csv"; covariate_columns=[:wt, :crcl])
```

!!! tip "Column names"
    Column names are case-insensitive. `ID`, `id`, and `Id` are all equivalent.
    Covariate columns are stored internally as lowercase symbols (`:wt`, `:crcl`),
    while model files reference them in uppercase (`WT`, `CRCL`).
