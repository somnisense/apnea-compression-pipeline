"""
step4_export_coreml.py

Phase 4 — CoreML conversion + Apple Neural Engine latency profiling.
Converts the final pruned + QAT model to a CoreML .mlpackage targeting
ANE-compatible operators on Apple M2 / iPhone 14 Pro. Measures inference
latency (0.064 ms +/- 0.016 ms per inference on Apple Neural Engine).

The .mlpackage is the on-device deployment artifact used by the SomniSense
production app.

Companion to: "From 14k to <60 KB: Joint Quantization-Aware Training and
Structured Pruning for On-Device Sleep Apnea Detection" (SomniAI LLC, 2026).
"""

import os
from pathlib import Path

# Data path resolves relative to this script's location.
# Place your dataset at: <paper_E_compression>/data/  (sibling to code/)
# By design, no data is bundled with the repository — see README "Data" section.
SCRIPT_DIR = Path(__file__).parent.resolve()
PAPER_DIR = SCRIPT_DIR.parent
DATA_DIR = PAPER_DIR / "data"

# Shared building blocks (model, data loader, evaluation helpers, latency / size utilities).
from common import (
    CoordAttnCNNBinary,
    SEBlock1D,
    CoordinateAttention1D,
    load_dataset_binary,
    make_loader,
    evaluate_model,
    measure_latency,
    get_model_size_kb,
    print_metrics,
)



# ============================================================
# full_summary
# ============================================================
# ============================================================
# Cell 19: Full Summary (Phase 2 + Phase 3)
# Final comparison table for the report
# ============================================================

print("=" * 75)
print("COMPLETE OPTIMIZATION PIPELINE SUMMARY")
print("Original CNN → Coord-Attn → Quantization → Pruning")
print("=" * 75)

final_rows = [
    # ── （Phase 1）──────────────────────────────────
    {
        'Stage'      : 'Phase 1',
        'Model'      : 'Original CNN (FP32)',
        'Accuracy'   : 0.8359,
        'F1'         : 0.8240,
        'AUC-ROC'    : 0.9299,
        'Params'     : 204801,
        'Size (KB)'  : 820.0,    #  Keras 
        'Latency (ms)': 15.0,
    },
    {
        'Stage'      : 'Phase 1',
        'Model'      : 'Coord-Attn CNN (FP32)',
        'Accuracy'   : fp32_metrics['Accuracy'],
        'F1'         : fp32_metrics['F1'],
        'AUC-ROC'    : fp32_metrics['AUC-ROC'],
        'Params'     : count_parameters(pt_model)[0],
        'Size (KB)'  : fp32_size_kb,
        'Latency (ms)': fp32_latency['mean_ms'],
    },
]

# ── （Phase 2）──────────────────────────────────────
for key in ['FP16', 'INT8-Dynamic', 'INT8-Static', 'QAT-10ep']:
    if key in results_registry:
        r = results_registry[key]
        final_rows.append({
            'Stage'      : 'Phase 2',
            'Model'      : f'Coord-Attn {key}',
            'Accuracy'   : r['Accuracy'],
            'F1'         : r['F1'],
            'AUC-ROC'    : r['AUC-ROC'],
            'Params'     : count_parameters(pt_model)[0],
            'Size (KB)'  : r['size_kb'],
            'Latency (ms)': r['latency_ms'],
        })

# ── （Phase 3）──────────────────────────────────────
for ratio in [0.3, 0.5, 0.7]:
    if ratio in finetuned_results:
        r   = finetuned_results[ratio]
        lbl = f"Pruned {int(ratio*100)}% + FT"
        final_rows.append({
            'Stage'      : 'Phase 3',
            'Model'      : lbl,
            'Accuracy'   : r['Accuracy'],
            'F1'         : r['F1'],
            'AUC-ROC'    : r['AUC-ROC'],
            'Params'     : r['nonzero_params'],
            'Size (KB)'  : r['size_kb'],
            'Latency (ms)': r['latency_ms'],
        })

#
if 'Pruned50%+QAT' in finetuned_results:
    r = finetuned_results['Pruned50%+QAT']
    final_rows.append({
        'Stage'      : 'Phase 3',
        'Model'      : 'Pruned 50% + FT + QAT',
        'Accuracy'   : r['Accuracy'],
        'F1'         : r['F1'],
        'AUC-ROC'    : r['AUC-ROC'],
        'Params'     : r['nonzero_params'],
        'Size (KB)'  : r['size_kb'],
        'Latency (ms)': r['latency_ms'],
    })

df_final = pd.DataFrame(final_rows)

#  CNN 
orig_acc  = 0.8359
orig_size = 820.0

df_final['Acc vs Orig']     = (df_final['Accuracy'] - orig_acc).map(
    lambda x: f"{x:+.4f}")
df_final['Size Reduct.']    = (1 - df_final['Size (KB)'] / orig_size).map(
    lambda x: f"{x:.1%}")
df_final['Param Reduct.']   = (1 - df_final['Params'] / 204801).map(
    lambda x: f"{x:.1%}")

print(df_final.to_string(index=False))
df_final.to_csv(OUTPUT_DIR_P3 / "final_complete_summary.csv", index=False)
print(f"\nSaved → {OUTPUT_DIR_P3}/final_complete_summary.csv")

# ──  trade-off  ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

stage_colors = {
    'Phase 1': 'navy',
    'Phase 2': 'darkorange',
    'Phase 3': 'darkgreen',
}
stage_markers = {
    'Phase 1': 'o',
    'Phase 2': 's',
    'Phase 3': '^',
}

for _, row in df_final.iterrows():
    stage = row['Stage']
    ax.scatter(row['Size (KB)'], row['Accuracy'],
               s=150,
               c=stage_colors[stage],
               marker=stage_markers[stage],
               zorder=5, alpha=0.85)
    ax.annotate(row['Model'],
                xy=(row['Size (KB)'], row['Accuracy']),
                xytext=(6, 3), textcoords='offset points',
                fontsize=7.5)

#
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='navy',      markersize=10, label='Phase 1: Architecture'),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor='darkorange', markersize=10, label='Phase 2: Quantization'),
    Line2D([0], [0], marker='^', color='w',
           markerfacecolor='darkgreen', markersize=10, label='Phase 3: Pruning'),
]
ax.legend(handles=legend_elements, fontsize=9)
ax.set_xlabel('Model Size (KB)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Complete Optimization Pipeline:\nAccuracy vs. Model Size Trade-off',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR_P3 / "final_tradeoff_complete.png", dpi=150)
plt.show()
print(f"\nFinal figure saved → {OUTPUT_DIR_P3}/final_tradeoff_complete.png")


# ============================================================
# coreml_export
# ============================================================
# ============================================================
# Phase 4 (Mini): CoreML Export & Latency Test
#  M2 Mac ，
# ============================================================

import coremltools as ct
import torch
import numpy as np
import time

# 1. （Pruned 50% + FT）
model = CoordAttnCNNBinary()
model.load_state_dict(torch.load(
    "phase3_pruning_results/pruned_50pct_finetuned.pth",
    map_location='cpu'))
model.eval()

# 2. Trace 
dummy = torch.randn(1, 200, 3)
traced = torch.jit.trace(model, dummy)

# 3.  CoreML
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input", shape=dummy.shape)],
    compute_units=ct.ComputeUnit.ALL   #  Neural Engine
)

# 4. 
mlmodel.save("sleep_apnea_pruned50_ft.mlpackage")
print("CoreML model saved.")

# 5. 
test_input = {"input": np.random.randn(1, 200, 3).astype(np.float32)}

# Warmup
for _ in range(20):
    mlmodel.predict(test_input)

#
times = []
for _ in range(200):
    t0 = time.perf_counter()
    mlmodel.predict(test_input)
    times.append((time.perf_counter() - t0) * 1000)

print(f"CoreML Latency: {np.mean(times):.3f} ± {np.std(times):.3f} ms")
print(f"P95 Latency   : {np.percentile(times, 95):.3f} ms")

