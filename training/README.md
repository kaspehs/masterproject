# Training

This folder contains the model training entrypoint, shared training utilities,
method-specific trainers, experiment configs, and cluster job scripts.

## Layout

`train.py`
: Main training entrypoint. It reads a YAML config and dispatches to the trainer
selected by the config `method` field.

`training_utils.py`
: Shared model, data-loading, rollout, plotting, and validation utilities.
This was previously named `HNN_helper.py`.

`architectures.py`
: Reusable neural-network architecture blocks used by the trainers.

`async_validate.py`
: Validation runner launched by trainers for asynchronous rollout validation.

`methods/`
: Method-specific training implementations:

- `correction/`
- `standalone/`

`configs/`
: YAML experiment configs. `configs/final_configs/` contains the delivery
configs; the top-level smoke configs are useful for quick checks. See
`configs/README.md` for the public config-key reference and thesis-section
cross-references.

`cluster/`
: Slurm/Olivia helper scripts for running training jobs.

## Running Training

From the repository root:

```bash
python -m training.train --config training/configs/correction_smoke.yml
```

Example delivery configs:

```bash
python -m training.train --config training/configs/final_configs/standalone_model.yml
python -m training.train --config training/configs/final_configs/frequency_correction_model.yml
python -m training.train --config training/configs/final_configs/force_correction_model.yml
python -m training.train --config training/configs/final_configs/combined_correction_model.yml
```

The configs expect training data under `vivana_cfd_data_pipeline/generated/`.
Build those datasets first with:

```bash
python vivana_cfd_data_pipeline/scripts/build_final_training_dataset.py
```

## Viewing Logs

Training writes TensorBoard event files under `logs/` by default. From the
repository root, start TensorBoard with:

```bash
tensorboard --logdir logs --port 6006
```
