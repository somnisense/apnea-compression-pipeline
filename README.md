# apnea-compression-pipeline

> **From a 14,001 FP32 Stage-2 classifier to a sub-60 KB INT8 model that runs at 0.064 ms latency on the Apple Neural Engine — without losing test accuracy.**

This is **Paper E** in a 3-paper series on smartphone-deployable sleep monitoring. The compressed model produced by this pipeline is the **Stage-2 classifier of a cascaded two-stage pipeline** introduced in companion work, after applying the Coord-Attn 1D architectural refinement also from companion work. Companion repositories:
- [`audio-sleep-cnn-baselines`](https://github.com/somnisense/audio-sleep-cnn-baselines) (Paper A — the cascaded two-stage pipeline that the Stage-2 model serves in)
- [`ca1d-sleep-apnea`](https://github.com/somnisense/ca1d-sleep-apnea) (Paper C — the Coord-Attn 1D Stage-2 architecture that this pipeline compresses)

A reproducible four-phase compression pipeline:

1. **Phase 1 — Architectural redesign**: Flatten + Dense head → GlobalAvgPool1D + Coord-Attn block + Dense head (204,801 → 14,001 parameters)
2. **Phase 2 — Quantization-Aware Training (QAT)** to INT8 on Linear layers
3. **Phase 3 — L1 structured filter pruning at 50%** with cosine-annealed fine-tuning
4. **Phase 4 — CoreML conversion and deployment** to Apple Neural Engine

The non-obvious finding: on a small dataset (n ≈ 3,000), aggressive QAT + structured pruning *improves* held-out test accuracy rather than degrading it. The compression acts as an implicit regularizer — consistent with what's been observed elsewhere in the lottery-ticket / pruning-as-regularization literature.

---

## Headline result

| Stage | Params | Size | Test accuracy | Test F1 |
|---|---|---|---|---|
| FP32 (pre-compression, Stage-2 = Coord-Attn 1D from Paper C) | 14,001 | ~60 KB | 87.14% | 86.94% |
| INT8 QAT | 14,001 | 14.6 KB | 87.21% | 87.03% |
| INT8 QAT + 50% L1 pruning | ~9,416 | **56.4 KB** | **88.49%** | **87.18%** |
| CoreML on Apple Neural Engine | — | — | (deploys cleanly) | **0.064 ms / inference** |

Accuracy goes *up* through both compression stages on this dataset. Paper §5 documents why this is consistent with regularization-on-small-data dynamics rather than an anomaly.

---

## Preprint

The full paper (English; results tables, per-seed metrics, figures, discussion) is available at:

- **Zenodo (canonical, citable DOI)**: [10.5281/zenodo.20663768](https://doi.org/10.5281/zenodo.20663768)
- **ORCID**: [0009-0002-4798-5161](https://orcid.org/0009-0002-4798-5161)
- **arXiv (cs.LG)**: *planned*

This repository is the **code companion** to the paper, not a mirror of it.

---

## Reproduce

```bash
cd code
pip install -r requirements.txt

# Phase 1 — TensorFlow / Keras: architectural redesign + FP32 baseline training
#   Replaces Flatten+Dense head with GlobalAvgPool1D + Coordinate-Attention 1D block.
#   Outputs a 14,001-parameter FP32 Keras model and 5-seed bootstrap metrics.
python step1_train_fp32.py

# Phase 2 — PyTorch: Quantization study
#   Trains a PyTorch FP32 baseline, then sweeps FP16 / INT8-Dynamic / INT8-Static /
#   per-layer sensitivity / QAT. QAT (10 epochs) selected as the production winner.
python step2_qat.py

# Phase 3 — PyTorch: L1-structured Conv1D pruning
#   Starts from the QAT checkpoint. L1 structured filter pruning at 30/50/70% with
#   20-epoch cosine-annealed fine-tune. 50% pruning is the production winner
#   (88.49% accuracy at 9,416 parameters).
python step3_prune.py

# Phase 4 — PyTorch + coremltools: CoreML export and ANE latency profiling
#   Converts the final pruned + QAT model to a .mlpackage and measures inference
#   latency on Apple Neural Engine (M2 / iPhone 14 Pro target).
python step4_export_coreml.py
```

`common.py` holds the shared PyTorch building blocks — the Coord-Attn 1D model
definition, the 200×3 feature-matrix data loader, and the per-event evaluation
metrics — imported by steps 2-4.

Phase 1 uses TensorFlow / Keras (matching the architectural redesign characterized
in companion work [`ca1d-sleep-apnea`](https://github.com/somnisense/ca1d-sleep-apnea)). Phases 2-4 use PyTorch +
coremltools, since the QAT primitives and the CoreML toolchain are more mature in
the PyTorch ecosystem. The two stacks share the same input feature contract
(200×3 acoustic matrices, where the third channel is the Stage-1 cascade output) but operate on independent model representations —
the FP32 checkpoint is exported and re-loaded into PyTorch at the Phase 1 /
Phase 2 boundary.

---

## What's in the CoreML output

The pipeline produces a CoreML `.mlpackage` that runs on the Apple Neural Engine on an iPhone with an A14 chip or later. Inference latency measured on iPhone 14 Pro: **0.064 ms / inference**.

The CoreML export step is fully reproducible from source — run `step4_export_coreml.py` against your own trained checkpoint to produce the deployment artifact. No `.mlpackage` binary is shipped in this repository.

---

## Patent notice

> The four-phase compression pipeline disclosed in this paper, the CA-1D Stage-2 architecture it operates on, the cascaded two-stage pipeline that surrounds it, and the system-level gating / event-triggered inference / privacy-preserving on-device architecture used in production deployment are the subject of **three co-filed U.S. provisional patent applications** by SomniAI LLC (filed 2026-06; application numbers pending). The paper and this repository disclose the QAT methodology, pruning protocol, and accuracy / latency / footprint trade-off measurements for reproducibility; production deployment specifics — particularly the system-level gating, event-driven triggering, and privacy-preserving on-device system architecture — are covered by the co-filed patent applications and are not described here.
>
> Code in this repository is licensed under MIT for research, evaluation, and reproducibility purposes (see [LICENSE](LICENSE)).

---

## Data

Uses the same 40-participant / 80-person-night audio-PSG-paired 200×3 feature-matrix dataset described in companion work — [`audio-sleep-cnn-baselines`](https://github.com/somnisense/audio-sleep-cnn-baselines) §3.1 and [`ca1d-sleep-apnea`](https://github.com/somnisense/ca1d-sleep-apnea) §3.2. **No data is redistributed in any of the three companion repos**, by design — the public release is an algorithm-framework only. The training and evaluation scripts run against any dataset that conforms to the 200×3 feature-matrix I/O contract (third channel = per-second Stage-1 snore-presence indicator from the cascade pipeline) described in the upstream READMEs.

---

## Citation

```bibtex
@misc{yang2026compression,
  author       = {Yang, L.},
  title        = {Compression Pipeline for Sub-Millisecond Sleep Apnea Detection
                  on Mobile Neural Engines},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20663768},
  howpublished = {Zenodo preprint, \url{https://doi.org/10.5281/zenodo.20663768}},
  note         = {Code: \url{https://github.com/somnisense/apnea-compression-pipeline}},
}
```

---

## License

Code: **MIT**. Patent rights are not granted by this license — see *Patent notice* above.

---

## About

Built and maintained by [**SomniAI LLC**](https://github.com/somnisense). The production app (SomniSense, Wellness category) that uses this pipeline runs on-device on **iOS and Android**: → **[somnisense.top](https://www.somnisense.top)**.

Research & validation hub: **[apneasense.com/research](https://apneasense.com/research)**.

Questions: `service@somnisense.top`.
