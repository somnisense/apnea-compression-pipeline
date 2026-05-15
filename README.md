# apnea-compression-pipeline

> **From 14,001 FP32 parameters to a sub-60 KB INT8 model that runs on the Apple Neural Engine — without losing test accuracy.**

A reproducible compression pipeline that takes the Coord-Attn 1D sleep apnea model from companion work and:

1. Applies **quantization-aware training (QAT)** to INT8
2. Applies **L1 structured filter pruning** at 50%
3. Converts the pruned-quantized model to **CoreML** for Apple Neural Engine deployment

The non-obvious finding: on a small dataset (n ≈ 3,000), aggressive QAT + structured pruning *improves* held-out test accuracy rather than degrading it. The compression acts as an implicit regularizer — consistent with what's been observed elsewhere in the lottery-ticket / pruning-as-regularization literature.

This repository accompanies the arXiv paper **"From 14k to <60 KB: Joint Quantization-Aware Training and Structured Pruning for On-Device Sleep Apnea Detection"** ([arXiv preprint](#) — link to be added on first upload).

---

## Headline result

| Stage | Params | Size | Test accuracy | Test F1 |
|---|---|---|---|---|
| FP32 (pre-compression) | 14,001 | 56.0 KB | 87.14% | 86.94% |
| INT8 QAT | 14,001 | 14.6 KB | 87.21% | 87.03% |
| INT8 QAT + 50% L1 pruning | ~7,000 | ~7.5 KB | **87.41%** | **87.18%** |
| CoreML on Apple Neural Engine | — | — | (deploys cleanly) | 0.064 ms / inference |

Accuracy goes *up* through both compression stages on this dataset. The paper §3 documents why this is consistent with regularization-on-small-data dynamics rather than an anomaly.

---

## Paper

The full paper (English, with all results tables, per-seed metrics, figures, and discussion) lives on arXiv. This repository is the **code companion** to that paper, not a mirror of it.

- **arXiv:** *(link to be added on first upload)*

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
#   (88.49% accuracy at 12,295 parameters).
python step3_prune.py

# Phase 4 — PyTorch + coremltools: CoreML export and ANE latency profiling
#   Converts the final pruned + QAT model to a .mlpackage and measures inference
#   latency on Apple Neural Engine (M2 / iPhone 14 Pro target).
python step4_export_coreml.py
```

`common.py` holds the shared PyTorch building blocks — the Coord-Attn 1D model
definition, the 200 × 3 feature-matrix data loader, and the per-event evaluation
metrics — imported by steps 2-4.

Phase 1 uses TensorFlow / Keras (matching the architectural redesign characterized
in the companion preprint on Coordinate-Attention 1D). Phases 2-4 use PyTorch +
coremltools, since the QAT primitives and the CoreML toolchain are more mature in
the PyTorch ecosystem. The two stacks share the same input feature contract
(200 × 3 acoustic matrices) but operate on independent model representations —
the FP32 checkpoint is exported and re-loaded into PyTorch at the Phase 1 /
Phase 2 boundary.

---

## What's in the CoreML output

The pipeline produces a CoreML `.mlpackage` that runs on the Apple Neural Engine on an iPhone with an A14 chip or later. Inference latency measured on iPhone 14 Pro: **0.064 ms / inference**.

The CoreML export step is fully reproducible from source — run `step4_export_coreml.py` against your own trained checkpoint to produce the deployment artifact. No `.mlpackage` binary is shipped in this repository.

---

## Patent notice

> The end-to-end deployment pipeline — including the audio pre-processing front-end that gates inference, the event-triggered inference scheduler, and certain implementation details of the CoreML export configuration — is the subject of a pending **US provisional patent application** filed by SomniAI LLC. This paper and repository disclose the QAT methodology, pruning protocol, and accuracy / latency / footprint trade-off measurements for reproducibility; production deployment specifics are not redistributed here.
>
> Code in this repository is licensed under MIT for research, evaluation, and reproducibility purposes (see [LICENSE](LICENSE)).

---

## Data

Uses the same 40-participant / 80-person-night audio-PSG-paired feature matrix dataset described in companion work — [`audio-sleep-cnn-baselines`](https://github.com/somnisense/audio-sleep-cnn-baselines) §3.1 and [`ca1d-sleep-apnea`](https://github.com/somnisense/ca1d-sleep-apnea) §3.2. **No data is redistributed in any of the three companion repos**, by design — the public release is an algorithm-framework only. The training and evaluation scripts run against any dataset that conforms to the 200×3 feature-matrix I/O contract described in the upstream READMEs.

---

## Citation

```bibtex
@misc{yang2026compression,
  author       = {Yang, L.},
  title        = {From {14k} to {<60 KB}: Joint Quantization-Aware Training and
                  Structured Pruning for On-Device Sleep Apnea Detection},
  year         = {2026},
  howpublished = {arXiv preprint},
  note         = {Code: \url{https://github.com/somnisense/apnea-compression-pipeline}},
}
```

---

## License

Code: **MIT**. Patent rights are not granted by this license — see *Patent notice* above.

---

## About

Built and maintained by [**SomniAI LLC**](https://github.com/somnisense). The production app that uses this pipeline runs on-device on **iOS and Android**: → **[somnisense.top](https://www.somnisense.top)**.

Questions: `service@somnisense.top`.
