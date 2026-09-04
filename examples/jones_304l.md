# Calibrating the Jones et al. 304L specimens

Six specimens: `xt10` and the bespoke `o1`, `o5`, `o14`, `o15`, `o18`.
Reference choices for each specimen are recorded in
`examples/jones_304l_specimens.yaml`, and the pipeline replays from
that record. The commands below show `o5` with its recorded values.

## Measurements

`data/jones_304l/` is untracked and holds, per specimen, a geometry
file and a DIC record:

| file | what it is |
| --- | --- |
| `O5-NominalGeometry.mat` | the outline and hole geometry the mesh is built from |
| `O5-4-Data.mat` | the DIC record: reference coordinates, per frame displacements, a per point `sigma` confidence, and load channels |

## 1. Choose the frame range

```
python examples/jones_304l_load_preview.py --data data/jones_304l/O5-4-Data.mat \
    --frame-min 3 --frame-max 700
```

Plots load against time and against extension with the candidate range
shaded, and writes a review figure of the excluded force artifacts
(`--drop-threshold`, `--exclude-frames`). Choose by eye: start where
the load is meaningfully positive, stop short of peak load, away from
failure.

## 2. Choose the cuts

```
python examples/jones_304l_cut_preview.py \
    --geometry data/jones_304l/O5-NominalGeometry.mat \
    --data data/jones_304l/O5-4-Data.mat \
    --y-min 40 --y-max 132 --frame-min 3 --frame-max 700
```

The specimen is longer than the region worth modeling, so it is cut at
two `y` values and the measured displacements drive those faces. A
point is valid where the record's per point `sigma` confidence reports
it; the figure shows the outline, the cuts, and the validity
intersected over the kept range, so the cuts sit where the data is
sound.

## 3. Choose the element size

```
python examples/jones_304l_mesh_preview.py \
    --geometry data/jones_304l/O5-NominalGeometry.mat \
    --y-min 40 --y-max 132 --h 3 --curvature-elements 0 12 20 28
```

Reports element count against boundary error at the tightest turns of
the holes. The
mesh refines by curvature toward `--curvature-elements` elements
around a full turn, leaving the rest of the face at `--h`.

## 4. Choose the region of interest band

```
python examples/jones_304l_roi_preview.py \
    --geometry data/jones_304l/O5-NominalGeometry.mat \
    --data data/jones_304l/O5-4-Data.mat \
    --mesh examples/meshes/jones_304l_o5_2d_y40_132_h3.msh \
    --y-min 40 --y-max 132 --frame-min 3 --frame-max 700 \
    --roi-band 1.4 1.6 1.8
```

One panel per candidate band. The band excludes a strip inward from
the free boundary, where the correlation window runs off the specimen
and the measurement stops short. The cut faces pass through the
interior of the measured field, so data reaches them without a band.

## 5. Record the choices

Add or update the specimen's entry in
`examples/jones_304l_specimens.yaml`: the frame range, the drop
threshold and any hand excluded frames, the two cut values, `h` and
the curvature elements, the region of interest band, and the number
of steps.

## 6. Generate meshes, archives, and input files

```
python examples/jones_304l_setup.py --specimens o5
```

Runs three stages with the recorded values: `meshes` generates the
meshes, `archives` converts the record to calibration data archives,
and `inputs` writes input files under `examples/jones_304l_inputs/o5/`.
`--steps` picks which stages run (`--steps inputs` writes only the
input files) and `--dims` which dimensions (`--dims 2` does only 2D).

The converter writes, stemmed on the mesh name in `data/jones_304l/`:

| file | what it is |
| --- | --- |
| `..._calibration_data.npz` | the archive: every usable frame remapped onto the stored nodes, the measured loads, the region of interest, and a constructed undeformed reference at t = 0 |
| `..._solve_times.txt` | the time schedule the solver steps through |
| `..._match_times.txt` | the times the objective matches the data at |

## 7. Schedules

```
python examples/jones_304l_schedule.py \
    --archive data/jones_304l/jones_304l_o5_2d_y40_132_h3_calibration_data.npz \
    --num-steps 100
```

Writes a new solve times and match times pair from the archive. The
default selection spreads the frames evenly in index over the kept
range; `--select force` spreads them evenly in load, and `--times`
takes a custom list. The two files start as copies of the same
selection. Solve times that are not match times contribute nothing to
the objective, and match times must be archive frame times. When a
step fails during a run, the solver splits its interval and inserts
solve times snapped to measured frames, while the match times stay
fixed.

## 8. Calibrate

```
python -m cmad.cli.main --devices 4 calibrate examples/jones_304l_inputs/o5/calibrate_2d.yaml
```

The generated calibrate files carry weights of 1.0 as placeholders.
Writes `active_params.json`, `opt_history.json`, `opt_status.json`,
`opt_params.yaml`, and the resolved input file under the output path.

## 9. Predict and compare

```
python -m cmad.cli.main --devices 4 primal examples/jones_304l_inputs/o5/primal_2d.yaml
python examples/jones_304l_compare.py \
    --deck examples/jones_304l_inputs/o5/calibrate_2d.yaml \
    --exodus results/jones_304l_o5/primal_2d/o5_primal_2d.exo \
    --reaction results/jones_304l_o5/primal_2d/reaction.csv \
    --out-dir results/jones_304l_o5/compare_2d
```

Edit the primal file's material values to whatever parameters you
want to inspect, such as a calibration's optimum. The primal writes
the solved field to exodus and the reaction series to a CSV; the
comparison writes `compare.exo` with `u_pred`, `u_meas`, and `u_diff`
on the stored nodes, a predicted against measured load figure, the
match terms per frame, the spatial rms of the mismatch, and a CSV of
the values.
