# Lesion-Focused Classification for Skin Cancer Detection

**Does background suppression help generalization?**
CS 289A (Spring 2026) final project — Tobias Roemer, Thomas Lee, Leo Li.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tnroemer/289-project/main?labpath=notebooks)

Skin-lesion classifiers trained end-to-end on dermoscopic images can exploit
non-lesion context (skin texture, hair, rulers, lighting artifacts) instead of
the lesion itself, which hurts when the imaging setup changes. This project runs
a **controlled, matched-conditions comparison** of full-image vs.
lesion-focused (background-suppressed) classification across four architectures,
trained on **HAM10000** and evaluated both in-distribution and
out-of-distribution on **PAD-UFES-20**. The task is binary `benign` vs
`malignant` over the labels overlapping with PAD (`akiec, bcc, bkl, mel, nv`).

The written report is in [`report/main.tex`](report/main.tex).

## Repository layout

| Path | What it is |
|------|------------|
| `src/data_setup/` | Download + label-normalize + split HAM10000 / PAD-UFES-20; build lesion-on-white images |
| `src/training/` | Shared HAM/PAD trainers + thin per-model wrappers + the DeepLabV3 segmenter |
| `src/evaluation/` | Cross-domain evaluation on PAD-UFES-20 and bootstrap CIs |
| `src/models/` | `build_model` (CNN / ViT / ResNet / pretrained ResNet-50) and the segmenter |
| `src/segmentation_unet.py` | Self-contained from-scratch U-Net pipeline behind the two notebooks |
| `src/data_setup/make_sample_data.py` | Generates the small synthetic sample dataset for the notebooks |
| `notebooks/` | `01_segmentation.ipynb`, `02_lesion_focused.ipynb` — runnable narrative of the lesion-focusing pipeline |
| `submit/` | SLURM scripts; `submit_full_pipeline.sh` chains the whole graph |
| `report/`, `proposal/`, `figures/` | Report sources and final figures |

## Setup

The project uses Python ≥ 3.11.

**Cluster / GPU pipeline** (the real experiments) — managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync           # creates .venv from pyproject.toml + uv.lock
export PYTHONPATH="$PWD/src"
```

**Notebooks / local / Binder** — a conservative CPU-only pinned set:

```bash
pip install -r requirements.txt
python -m data_setup.make_sample_data    # writes the synthetic sample to data/sample/
```

## Running the key experiments

The canonical dependency graph is `submit/submit_full_pipeline.sh`. Run modules
as `python -m <module>` with `src/` on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src"

python -m data_setup.create_data                     # download + stratified splits (seed 42)
python -m training.train_ham10000_segmentation_model # DeepLabV3-ResNet50 segmenter
python -m data_setup.create_lesion_white_data        # masks -> lesion-on-white images
python -m training.train_ham10000_full_image_resnet  # one of 8 (model x image_source) wrappers
python -m evaluation.evaluate_pad_ufes20_full_image_models
python -m evaluation.bootstrap                       # 95% bootstrap CIs from preds/*.csv
```

On PSC Bridges-2: `sbatch submit/submit_full_pipeline.sh` (jobs chained with
`--dependency=afterok`). Many paths are hardcoded to
`/ocean/projects/mth250011p/...`; running end-to-end elsewhere requires editing
the `DATA_ROOT` / `RUN_DIR` constants at the top of the scripts named in
[`CLAUDE.md`](CLAUDE.md). W&B logging is enabled (project `skin-cancer-cnn`).

**Key conventions:** binary classification with one logit + `BCEWithLogitsLoss`
(`pos_weight = #benign/#malignant`); model selection by **highest validation
specificity at sensitivity ≥ 0.90**; the chosen threshold is stored on the
checkpoint and reused at test time; `seed = 42` everywhere.

## Notebooks

`notebooks/01_segmentation.ipynb` trains the from-scratch U-Net lesion
segmenter; `notebooks/02_lesion_focused.ipynb` loads it and produces
background-suppressed lesion-focused images. Run them **in order** (01 writes
the checkpoint that 02 reads).

Both default to `SAMPLE = True` and use the small synthetic dataset in
`data/sample/`, so they run top-to-bottom in seconds on CPU with no kagglehub
credentials — this is what Binder and the website use. Set `SAMPLE = False`
(top of each notebook) to point them at the real HAM10000 / ISIC-2018 data on a
machine that has it.

Reproducibility check: `pip install -r requirements.txt`, then
`python -m data_setup.make_sample_data`, then *Restart Kernel & Run All* on
`01` then `02`.

## Website (MyST)

The repo is configured to build a [MyST](https://mystmd.org/) site from
`index.md` plus the two executed notebooks:

```bash
npm install -g mystmd
python -m data_setup.make_sample_data
myst start            # local preview
myst build --html --execute
```

`.github/workflows/deploy-myst.yml` builds and deploys this site to GitHub Pages
on every push to `main` (enable Pages → "GitHub Actions" in repo settings).

## Citation of adapted code

`torch`, `torchvision`, and `kagglehub` are used for data loading, standard
architectures, and training utilities. Pretrained ResNet-50 / ViT / DeepLabV3
weights come from `torchvision`. All training, segmentation, threshold
selection, lesion-focusing, evaluation, and bootstrap logic is written by the
team. Datasets: HAM10000 (Tschandl et al.), ISIC 2018 Task 1 (Codella et al.),
PAD-UFES-20 (Pacheco et al.) — see `report/references.bib`.
