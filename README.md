# Masterproject

This repository contains the code used for the master thesis experiments on
data-assisted VIVANA-TD correction models for cross-flow vortex-induced
vibrations. It includes the CFD data pipeline, the VIVANA-TD replay code,
training code for correction and standalone neural models, plotting utilities,
and tests.

The thesis LaTeX repository may be cloned locally as `Master_Thesis/` for
reference. That folder is ignored by git and is not part of this code delivery.

## Repository Layout

`training/`
: Model training code. The main entrypoint is `training/train.py`, with
method-specific trainers under `training/methods/`. See `training/README.md`.

`training/configs/`
: YAML experiment configs. The top-level smoke configs are for quick checks;
`final_configs/` contains the final correction and standalone model configs.
See `training/configs/README.md` for the config-key reference.

`vivana_cfd_data_pipeline/`
: CFD preprocessing, VIVANA-TD replay, dataset generation, analysis, and plotting
for building the training datasets. See `vivana_cfd_data_pipeline/README.md`.

`core/`
: Shared runtime, logging, optimizer, and learning-rate scheduling helpers used
by the training code.

`plotting_etc/`
: Standalone plotting and evaluation scripts for trained models, TensorBoard
summaries, leave-one-\(U_r\)-out results, rollout timeseries, and timestep
sensitivity diagnostics.

`tests/`
: Unit and smoke tests for config parsing, rollout losses, VIVANA-TD helpers,
and model output constraints.

`figs/`
: Generated figures and diagnostics. This folder is ignored by git.

`logs/`
: TensorBoard logs and validation outputs from training runs. This folder is
ignored by git.

`models/`
: Saved model checkpoints. This folder is ignored by git.

`requirements_olivia.txt`
: Minimal dependency list used for the Olivia/cluster environment.

## Typical Workflow

Build or refresh the final dataset:

```bash
python vivana_cfd_data_pipeline/scripts/export_cfd_to_npz.py
python vivana_cfd_data_pipeline/scripts/build_final_training_dataset.py
```

Run a smoke training config:

```bash
python -m training.train --config training/configs/correction_smoke.yml
python -m training.train --config training/configs/standalone_smoke.yml
```

Run a final model config:

```bash
python -m training.train --config training/configs/final_configs/force_correction_model.yml
python -m training.train --config training/configs/final_configs/frequency_correction_model.yml
python -m training.train --config training/configs/final_configs/combined_correction_model.yml
python -m training.train --config training/configs/final_configs/standalone_model.yml
```

View training logs:

```bash
tensorboard --logdir logs --port 6006
```

Run tests:

```bash
python -m unittest discover -s tests
```

Use Python 3.11 or newer for the current codebase.

## Version-Control Notes

Generated data, logs, model checkpoints, plots, raw CFD files, and the local
thesis checkout are intentionally ignored. The versioned source of truth is the
code, metadata, configs, and documentation needed to rebuild datasets and rerun
the experiments.
