"""
    create_dataset(ids, dose_amt, obs_times; cmt=1, rate=0.0) -> DataFrame

Build a NONMEM-format template DataFrame with one bolus dose row (EVID=1) and one
observation slot per time point (EVID=0, DV=0.0) for each subject ID.

After calling this function you can add per-subject covariate columns and then
call `simulate()` to fill in the `DV` column with simulated observations.
"""
function create_dataset(ids, dose_amt, obs_times; cmt=1, rate=0.0)
    rows = []
    for id in ids
        push!(rows, (ID=id, TIME=0.0, AMT=Float64(dose_amt), DV=missing,
                     EVID=1, MDV=1, CMT=cmt, RATE=Float64(rate)))
        for t in obs_times
            push!(rows, (ID=id, TIME=Float64(t), AMT=missing, DV=0.0,
                         EVID=0, MDV=0, CMT=cmt, RATE=Float64(rate)))
        end
    end
    return DataFrame(rows)
end
