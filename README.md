# Trilobyte

This repository contains the research experiments for **Trilobyte** — a language-model-based lossless audio codec. It includes baseline comparisons, training experiments, and paper figures.

---

## Repository layout

- `baselines/` — Baseline experiments, eval pipelines, NAC codecs
- `trilobyte/` — Training code, configs, model experiments
- `paper_figures/` — Scripts to reproduce paper tables and figures

---

## Try Trilobyte

The official Trilobyte lossless codec — including our **final trained model** — is available at:

**[https://github.com/pnlong/trilobyte-lossless-codec](https://github.com/pnlong/trilobyte-lossless-codec)**

If you just want to compress or decompress audio with Trilobyte, go there. This repository is for research, training, and experiments.

---

## Quick start

- **Encode/decode audio:** See [trilobyte-lossless-codec](https://github.com/pnlong/trilobyte-lossless-codec).
- **Run FLAC baseline:** See `baselines/flac_eval/`.
- **Run LMIC (in-context) eval:** See `baselines/in_context_eval/`.
- **NAC codec experiments:** See `baselines/nac/`.
- **Reproduce paper figures:** See `paper_figures/`.
