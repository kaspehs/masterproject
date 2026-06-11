# Vivana CFD Data Pipeline

This folder contains the CFD data handling, Vivana-TD replay implementation,
dataset generation scripts, and analysis utilities used to build the training
datasets for the master thesis experiments.

## Folder Layout

`raw/`
: Raw CFD `.dog` files. This folder is ignored by git because the files are
large source data.

`metadata/`
: Versioned metadata and cleaning manifests for the raw CFD cases. These files
document which raw cases are kept, trimmed, or dropped and provide physical
parameters used during export.

`generated/`
: Generated `.npz` datasets. This folder is ignored by git. Important generated
subfolders are:

- `cfd_npz_exports/`: cleaned CFD time series exported from the raw `.dog` files.
- `final_dataset/`: final `train/` and `val_seen/` dataset produced by
  `scripts/build_final_training_dataset.py`.
- `loo_ur_td_burnin_trimmed_final/`: optional leave-one-reduced-velocity-out
  datasets.

`outputs/`
: Analysis outputs, diagnostic plots, and intermediate reports. This folder is
ignored by git.

`scripts/`
: Runnable scripts that produce data or perform a specific pipeline step.
The main delivery script is `build_final_training_dataset.py`.

`helpers/`
: Shared helper modules used by scripts, analysis, plotting, and model rollout
evaluation. These are not meant to be run directly.

`analysis/`
: Analysis scripts for CFD/Vivana-TD comparison and parameter optimization.

`plotting/`
: Plot-specific scripts used for figures and diagnostics.

`notebooks/`
: Exploratory notebooks retained for traceability. Prefer the scripts for
reproducible runs.

`vivana_td/`
: Python implementation of the Vivana-TD model and hidden-state replay used by
the dataset pipeline.

## Rebuilding The Final Dataset

From the repository root:

```bash
python vivana_cfd_data_pipeline/scripts/export_cfd_to_npz.py
python vivana_cfd_data_pipeline/scripts/build_final_training_dataset.py
```

The first command exports cleaned CFD `.npz` files into
`vivana_cfd_data_pipeline/generated/cfd_npz_exports/`.

The second command builds the final training dataset from those `.npz` files,
including Vivana-TD burn-in trimming, `train/` and `val_seen/` splits, surrogate
validation points, and a dataset manifest.

Useful options:

```bash
python vivana_cfd_data_pipeline/scripts/build_final_training_dataset.py --no-overwrite
python vivana_cfd_data_pipeline/scripts/build_final_training_dataset.py --no-burnin-diagnostics
python vivana_cfd_data_pipeline/scripts/build_final_training_dataset.py --exclude-ur 5.75
python vivana_cfd_data_pipeline/scripts/build_final_training_dataset.py --loo
```

`--exclude-ur` removes the matching label reduced velocity from both `train/`
and `val_seen`. The same value is automatically excluded from the surrogate
validation anchor set so the surrogate targets are generated only from retained
training reduced velocities. If no reduced velocity is excluded from the final
dataset, no reduced velocity is excluded from the surrogate anchors.

## Delivery Notes

Keep `metadata/`, `scripts/`, `helpers/`, `analysis/`, `plotting/`, and
`vivana_td/` under version control. Keep `raw/`, `generated/`, and `outputs/`
out of git unless a specific small artifact is intentionally needed for the
thesis delivery.

Many pipeline configuration options are defined as constants near the top of
the two rebuild scripts listed above. `scripts/export_cfd_to_npz.py` controls
the raw CFD export and cleaning choices, while
`scripts/build_final_training_dataset.py` controls the final dataset build,
including Vivana-TD coefficients, TD memory settings, phase wrapping, and the
force phase convention.
