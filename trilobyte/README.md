# Trilobyte — Main Implementation

This repository is the **main implementation** of **Trilobyte**: lossless neural audio compression. It contains training code for the models (T5-based stereo prediction, GPT-2 byte-level), tokenization (BPE and raw bytes), dataset metadata, and experiment configs.

---

## Related repositories

| Repo | Description | Link |
|------|-------------|------|
| **Experiments / sandbox** | LNAC experiments, FLAC/LMIC eval, lossless codec prototypes (LDAC, LEC, LNAC), perceptual dithering, etc. | [LNAC (ZacharyNovack/lnac)](https://github.com/ZacharyNovack/lnac) |
| **Official lossless codec** | Standalone Trilobyte encode/decode library (CLI + API). | [trilobyte-lossless-codec](https://github.com/pnlong/trilobyte-lossless-codec) |

---

## Layout

```
trilobyte/
  train.py                  # T5-based stereo channel prediction training
  train_gpt2.py             # GPT-2 byte-level audio training
  splus.py                  # S+ optimizer
  environment.yml           # Conda environment
  parse_birdvox.sh          # Birdvox data prep script
  notebooks/
    Hierarchical_PCM_modeling.ipynb
    playground.ipynb
  configs/
    runs/                   # Training run/experiment configs
      mega_run_data.json
      scale_run_data.json
    dataset_info/           # Per-dataset file metadata (46 JSONs)
      *_info.json           # filename -> {length, sample_rate, bits_per_sample, n_channels}
      *_lengths*.json       # MusDB18 length manifests
      new_train_files.json  # Multi-dataset train split
      new_val_files.json    # Multi-dataset val split
```

---

## Root-level Python files

| File | Purpose |
|------|---------|
| `train.py` | T5-based seq2seq training for **stereo channel prediction** (left -> right). Uses PyTorch Lightning, loads stereo WAVs as fixed-length chunks, maps 16-bit PCM to token ids (vocab 65536), trains T5 (Hugging Face) with optional WandB logging and per-index loss/bpb. |
| `train_gpt2.py` | **GPT-2** training for **byte-level audio**: loads audio as byte tokens, trains autoregressive next-token prediction. Used for models that predict LSBs from MSBs or full byte streams for compression. Includes mu-law encode, `load_audio_raw` / `load_audio_mulaw`, and `split_to_bytes` (PCM -> interleaved byte tokens). |
| `splus.py` | **S+ optimizer**: custom PyTorch optimizer (Adam-like with EMA, inverse every N steps, weight decay). |
| `parse_birdvox.sh` | Shell script to fix/split Birdvox FLACs (e.g. segment or re-encode with ffmpeg). |

---

## `configs/`

### `configs/runs/`

Experiment configs driving multi-dataset or multi-run training without hardcoding paths in Python. Each JSON has a `datasets` key mapping dataset name -> `{train_data_dir, val_data_dir, train_metadata_path, val_metadata_path, sample_rate, stereo_interleave, ...}`.

| File | Description |
|---|---|
| `mega_run_data.json` | Full multi-dataset training run config |
| `scale_run_data.json` | Scaling experiment run config |

### `configs/dataset_info/`

Dataset manifests used by `train.py` / `train_gpt2.py` (`AudioByteDataset`) to discover audio files and read correct format. Datasets covered: beethoven, birdvox, epidemic, librispeech, ljspeech, musdb18stereo, sc09, torrent, trilobyte, vctk, youtube_mix.

- **`*_train_info.json` / `*_val_info.json`** — per-dataset train/val split metadata: `filename -> {length, sample_rate, bits_per_sample, n_channels}`
- **`new_train_files.json` / `new_val_files.json`** — multi-dataset split lists spanning torr, ljspeech, vctk, birdvox, epidemic, sc09, beethoven, youtube_mix
- **`musdbstereo_lengths*.json`** — MusDB18 stereo clip lengths for chunking
- **`musdbstereo_valid_mix_info.json`** — MusDB18 valid-mix file metadata

Paths in configs point at machine-specific dirs (e.g. `graft*`, `arrakis`); adjust for your environment or generate new manifests for your data layout.

---

## Quick start

- **Train T5 stereo model:** `python train.py --data_dir <dir> --chunk_size <n> --batch_size <n> --project <wandb_project>`
- **Train GPT-2 byte-level:** `python train_gpt2.py --metadata_path configs/dataset_info/<dataset>_train_info.json`
- **Encode/decode audio:** See the [trilobyte-lossless-codec](https://github.com/pnlong/trilobyte-lossless-codec) repo.
