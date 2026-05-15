"""
step3_prune.py

Phase 3 — L1-structured Conv1D filter pruning + cosine-annealed
fine-tune (20 epochs). Starts from the QAT-finalized model produced by
step2_qat.py. Evaluates pruning at 30% / 50% / 70% ratios; 50% pruning
is selected as the production winner (88.49% accuracy at 12,295 params).

Outputs a pruned + fine-tuned INT8 checkpoint consumed by step4_export_coreml.py.

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
# phase_intro
# ============================================================
# ============================================================
# Phase 3: Structured Pruning
# Cell 13: Pruning Utilities
# ============================================================

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

OUTPUT_DIR_P3 = Path("phase3_pruning_results")
OUTPUT_DIR_P3.mkdir(exist_ok=True)

# ──  ──────────────────────────────────────────────────

def count_parameters(model):
    """。"""
    total  = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    return total, nonzero


def count_zero_ratio(model):
    """（）。"""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            weight = module.weight.data
            total  = weight.numel()
            zeros  = (weight == 0).sum().item()
            stats[name] = {
                'total'      : total,
                'zeros'      : zeros,
                'sparsity'   : zeros / total,
                'shape'      : list(weight.shape),
            }
    return stats


def get_model_size_kb_p3(model):
    """。"""
    tmp = OUTPUT_DIR_P3 / "_tmp.pth"
    torch.save(model.state_dict(), tmp)
    size = tmp.stat().st_size / 1024
    tmp.unlink()
    return size


def print_sparsity_table(sparsity_stats):
    """。"""
    print(f"\n{'Layer':<25} {'Shape':<20} {'Total':>8} "
          f"{'Zeros':>8} {'Sparsity':>10}")
    print("-" * 75)
    for name, s in sparsity_stats.items():
        print(f"{name:<25} {str(s['shape']):<20} "
              f"{s['total']:>8,} {s['zeros']:>8,} "
              f"{s['sparsity']:>9.1%}")


# ============================================================
# l1_prune
# ============================================================
# ============================================================
# Cell 14: L1 Structured Filter Pruning on Conv1D Layers
# ============================================================

def apply_structured_pruning(model, prune_ratio):
    """
     Conv1D  L1-norm 。
    
    prune_ratio:  filter  (0.0 ~ 1.0)
    ：prune.ln_structured （ mask），
           make_permanent 。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            prune.ln_structured(
                module,
                name   = 'weight',
                amount = prune_ratio,
                n      = 1,          # L1 norm
                dim    = 0           # dim=0 →  output filter 
            )
    return model


def make_pruning_permanent(model):
    """
     mask （ weight_orig  weight_mask，
    ），。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
    return model


def load_fresh_model(weights_path):
    """ FP32 。"""
    m = CoordAttnCNNBinary().to(DEVICE)
    m.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return m


# ── 14A:  ─────────────────────────────────────────────
PRUNE_RATIOS = [0.0, 0.3, 0.5, 0.7]

# FP32 （ratio=0.0） Phase 2 
pruning_results = {}
pruning_results[0.0] = {
    'Accuracy'   : fp32_metrics['Accuracy'],
    'Precision'  : fp32_metrics['Precision'],
    'Recall'     : fp32_metrics['Recall'],
    'F1'         : fp32_metrics['F1'],
    'AUC-ROC'    : fp32_metrics['AUC-ROC'],
    'Specificity': fp32_metrics['Specificity'],
    'size_kb'    : fp32_size_kb,
    'latency_ms' : fp32_latency['mean_ms'],
    'p95_ms'     : fp32_latency['p95_ms'],
    'total_params'  : count_parameters(pt_model)[0],
    'nonzero_params': count_parameters(pt_model)[1],
    'sparsity'      : 0.0,
}

print("=" * 60)
print("PHASE 3: L1 Structured Filter Pruning")
print("=" * 60)

# ── 14B:  ───────────────────────────────────────
for ratio in PRUNE_RATIOS:
    if ratio == 0.0:
        continue

    print(f"\n{'─'*50}")
    print(f"Pruning ratio: {ratio:.0%}")
    print(f"{'─'*50}")

    # 1. 
    model_pruned = load_fresh_model(SAVE_PATH)
    model_pruned.eval()

    # 2. （ mask）
    apply_structured_pruning(model_pruned, ratio)

    # 3. 
    stats = count_zero_ratio(model_pruned)
    print_sparsity_table(stats)

    # 4.  mask
    make_pruning_permanent(model_pruned)

    # 5. （，）
    metrics, _, _ = evaluate_model(model_pruned, test_loader)
    latency       = measure_latency(model_pruned)
    total_p, nz_p = count_parameters(model_pruned)

    save_path = OUTPUT_DIR_P3 / f"pruned_{int(ratio*100)}pct.pth"
    torch.save(model_pruned.state_dict(), save_path)
    size_kb = get_model_size_kb_p3(model_pruned)

    pruning_results[ratio] = {
        **metrics,
        'size_kb'       : size_kb,
        'latency_ms'    : latency['mean_ms'],
        'p95_ms'        : latency['p95_ms'],
        'total_params'  : total_p,
        'nonzero_params': nz_p,
        'sparsity'      : 1 - nz_p / total_p,
    }

    print(f"\n  ► Accuracy   : {metrics['Accuracy']:.4f}  "
          f"(Δ vs FP32: {metrics['Accuracy']-fp32_metrics['Accuracy']:+.4f})")
    print(f"  ► F1         : {metrics['F1']:.4f}")
    print(f"  ► Params     : {total_p:,}  (nonzero: {nz_p:,})")
    print(f"  ► Size       : {size_kb:.2f} KB")
    print(f"  ► Latency    : {latency['mean_ms']:.3f} ms")


print("\n✅ Cell 14 （）")


# ============================================================
# finetune
# ============================================================
# ============================================================
# Cell 15: Fine-tuning After Pruning
# ============================================================

def finetune_pruned_model(pruned_model, train_loader, val_loader,
                          epochs=20, lr=5e-4):
    """
    。
    ，。
    """
    optimizer = torch.optim.Adam(
        pruned_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.BCELoss()

    best_val_acc  = 0.0
    best_state    = None

    for epoch in range(1, epochs + 1):
        # ── Train ──
        pruned_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(pruned_model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ── Validate ──
        if epoch % 5 == 0 or epoch == epochs:
            pruned_model.eval()
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = pruned_model(xb.to(DEVICE))
                    correct += ((pred > 0.5).float() ==
                                yb.to(DEVICE)).sum().item()
            val_acc = correct / len(val_loader.dataset)
            lr_now  = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:>3d}/{epochs} | "
                  f"val_acc={val_acc:.4f} | lr={lr_now:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = deepcopy(pruned_model.state_dict())

    if best_state:
        pruned_model.load_state_dict(best_state)
        print(f"  Best val_acc={best_val_acc:.4f} loaded.")

    return pruned_model


# ──  ─────────────────────────────────────────────────────
FINETUNE_EPOCHS = 20   # ：10 / 20 / 30
finetuned_results = {}

# FP32 
finetuned_results[0.0] = pruning_results[0.0]

print("=" * 60)
print("PHASE 3: Fine-tuning After Pruning")
print("=" * 60)

for ratio in PRUNE_RATIOS:
    if ratio == 0.0:
        continue

    print(f"\n{'─'*50}")
    print(f"Fine-tuning pruned model (ratio={ratio:.0%})")
    print(f"{'─'*50}")

    # 1.  FP32  → （）
    model_ft = load_fresh_model(SAVE_PATH)
    apply_structured_pruning(model_ft, ratio)
    make_pruning_permanent(model_ft)

    # 2. 
    model_ft = finetune_pruned_model(
        model_ft, train_loader, val_loader,
        epochs=FINETUNE_EPOCHS, lr=5e-4)

    # 3. 
    model_ft.eval()
    metrics, _, _ = evaluate_model(model_ft, test_loader)
    latency       = measure_latency(model_ft)
    total_p, nz_p = count_parameters(model_ft)

    save_path = OUTPUT_DIR_P3 / f"pruned_{int(ratio*100)}pct_finetuned.pth"
    torch.save(model_ft.state_dict(), save_path)
    size_kb = get_model_size_kb_p3(model_ft)

    finetuned_results[ratio] = {
        **metrics,
        'size_kb'       : size_kb,
        'latency_ms'    : latency['mean_ms'],
        'p95_ms'        : latency['p95_ms'],
        'total_params'  : total_p,
        'nonzero_params': nz_p,
        'sparsity'      : 1 - nz_p / total_p,
    }

    before_acc = pruning_results[ratio]['Accuracy']
    after_acc  = metrics['Accuracy']
    print(f"\n  ► Before FT  : {before_acc:.4f}")
    print(f"  ► After  FT  : {after_acc:.4f}  "
          f"(recovery: {after_acc - before_acc:+.4f})")
    print(f"  ► vs FP32    : {after_acc - fp32_metrics['Accuracy']:+.4f}")
    print(f"  ► F1         : {metrics['F1']:.4f}")
    print(f"  ► Size       : {size_kb:.2f} KB")
    print(f"  ► Latency    : {latency['mean_ms']:.3f} ms")

print("\n✅ Cell 15 （ + ）")


# ============================================================
# prune_qat_combined
# ============================================================
# ============================================================
# Cell 16: Pruning + QAT Combined (Ultimate Compression)
#  QAT
# ============================================================

#
#  50%（ Cell 15 ）
BEST_PRUNE_RATIO = 0.5

print("=" * 60)
print(f"PHASE 3: Pruning({int(BEST_PRUNE_RATIO*100)}%) + QAT Combined")
print("=" * 60)

# ── 16A: + ───────────────────────────────────
model_combined = load_fresh_model(SAVE_PATH)
apply_structured_pruning(model_combined, BEST_PRUNE_RATIO)
make_pruning_permanent(model_combined)

#
model_combined = finetune_pruned_model(
    model_combined, train_loader, val_loader,
    epochs=FINETUNE_EPOCHS, lr=5e-4)
model_combined.eval()

# （）
metrics_before_qat, _, _ = evaluate_model(model_combined, test_loader)
print(f"\nPruned+FT baseline accuracy: "
      f"{metrics_before_qat['Accuracy']:.4f}")

# ── 16B:  QAT ──────────────────────────────
# + QAT 
pruned_ft_path = OUTPUT_DIR_P3 / "pruned_50pct_finetuned_for_qat.pth"
torch.save(model_combined.state_dict(), pruned_ft_path)

class QATLinearOnlyFromPruned(QATLinearOnlyCNN):
    """
     QATLinearOnlyCNN，
     FP32 。
    """
    def load_from_pruned(self, pruned_state_dict):
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in pruned_state_dict.items():
            if name in own_state and \
               own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                skipped += 1
        print(f"  Loaded {loaded} layers, skipped {skipped} layers.")


print("\nRunning QAT on pruned model (10 epochs)...")

pruned_qat_model = QATLinearOnlyFromPruned().to(DEVICE)
pruned_qat_model.load_from_pruned(
    torch.load(pruned_ft_path, map_location=DEVICE))

#  qconfig
set_qat_qconfig_linear_only(pruned_qat_model, QENGINE)
tq.prepare_qat(pruned_qat_model, inplace=True)

# QAT 
optimizer  = torch.optim.Adam(
    pruned_qat_model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion  = nn.BCELoss()

pruned_qat_model.train()
for epoch in range(1, 11):   # 10 epochs
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(pruned_qat_model(xb), yb)
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        pruned_qat_model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = pruned_qat_model(xb.to(DEVICE))
                correct += ((pred > 0.5).float() ==
                            yb.to(DEVICE)).sum().item()
        val_acc = correct / len(val_loader.dataset)
        print(f"  QAT Epoch {epoch}/10 | val_acc={val_acc:.4f}")
        pruned_qat_model.train()

#  INT8
pruned_qat_model.eval()
tq.convert(pruned_qat_model, inplace=True)

# ── 16C:  ─────────────────────────────────────────
combined_metrics, _, _ = evaluate_model(pruned_qat_model, test_loader)
combined_latency        = measure_latency(pruned_qat_model)

combined_save = OUTPUT_DIR_P3 / "pruned_50pct_qat.pth"
torch.save(pruned_qat_model.state_dict(), combined_save)
combined_size_kb = get_model_size_kb_p3(pruned_qat_model)

print(f"\n{'─'*50}")
print(f"Pruning({int(BEST_PRUNE_RATIO*100)}%) + QAT Results:")
print(f"{'─'*50}")
print(f"  Accuracy  : {combined_metrics['Accuracy']:.4f}  "
      f"(vs FP32: {combined_metrics['Accuracy']-fp32_metrics['Accuracy']:+.4f})")
print(f"  F1        : {combined_metrics['F1']:.4f}")
print(f"  AUC-ROC   : {combined_metrics['AUC-ROC']:.4f}")
print(f"  Size      : {combined_size_kb:.2f} KB  "
      f"(vs FP32: {1-combined_size_kb/fp32_size_kb:.1%} reduction)")
print(f"  Latency   : {combined_latency['mean_ms']:.3f} ms")

#
finetuned_results['Pruned50%+QAT'] = {
    **combined_metrics,
    'size_kb'       : combined_size_kb,
    'latency_ms'    : combined_latency['mean_ms'],
    'p95_ms'        : combined_latency['p95_ms'],
    'total_params'  : count_parameters(pruned_qat_model)[0],
    'nonzero_params': count_parameters(pruned_qat_model)[1],
    'sparsity'      : BEST_PRUNE_RATIO,
}

print("\n✅ Cell 16 ")


# ============================================================
# summary_table
# ============================================================
# ============================================================
# Cell 17: Phase 3 Summary Table
# ============================================================

print("=" * 70)
print("PHASE 3 PRUNING SUMMARY")
print("=" * 70)

# ── 17A:  DataFrame ───────────────────────────────────
rows = []

#  FP32（）
rows.append({
    'Model'         : 'FP32 (No Pruning)',
    'Prune Ratio'   : '0%',
    'Fine-tuned'    : '—',
    'Accuracy'      : fp32_metrics['Accuracy'],
    'F1'            : fp32_metrics['F1'],
    'AUC-ROC'       : fp32_metrics['AUC-ROC'],
    'Params'        : count_parameters(pt_model)[0],
    'Size (KB)'     : fp32_size_kb,
    'Latency (ms)'  : fp32_latency['mean_ms'],
})

# （ vs ）
for ratio in [0.3, 0.5, 0.7]:
    label = f"{int(ratio*100)}%"

    #
    if ratio in pruning_results:
        r = pruning_results[ratio]
        rows.append({
            'Model'       : f'Pruned {label}',
            'Prune Ratio' : label,
            'Fine-tuned'  : 'No',
            'Accuracy'    : r['Accuracy'],
            'F1'          : r['F1'],
            'AUC-ROC'     : r['AUC-ROC'],
            'Params'      : r['nonzero_params'],
            'Size (KB)'   : r['size_kb'],
            'Latency (ms)': r['latency_ms'],
        })

    #
    if ratio in finetuned_results:
        r = finetuned_results[ratio]
        rows.append({
            'Model'       : f'Pruned {label} + FT',
            'Prune Ratio' : label,
            'Fine-tuned'  : 'Yes',
            'Accuracy'    : r['Accuracy'],
            'F1'          : r['F1'],
            'AUC-ROC'     : r['AUC-ROC'],
            'Params'      : r['nonzero_params'],
            'Size (KB)'   : r['size_kb'],
            'Latency (ms)': r['latency_ms'],
        })

#
if 'Pruned50%+QAT' in finetuned_results:
    r = finetuned_results['Pruned50%+QAT']
    rows.append({
        'Model'       : 'Pruned 50% + FT + QAT',
        'Prune Ratio' : '50%',
        'Fine-tuned'  : 'Yes+QAT',
        'Accuracy'    : r['Accuracy'],
        'F1'          : r['F1'],
        'AUC-ROC'     : r['AUC-ROC'],
        'Params'      : r['nonzero_params'],
        'Size (KB)'   : r['size_kb'],
        'Latency (ms)': r['latency_ms'],
    })

df_p3 = pd.DataFrame(rows)

#  FP32 
fp32_acc_ref  = fp32_metrics['Accuracy']
fp32_size_ref = fp32_size_kb

df_p3['Acc Δ']        = (df_p3['Accuracy'] - fp32_acc_ref).round(4)
df_p3['Size Reduct.'] = (1 - df_p3['Size (KB)'] / fp32_size_ref).map(
    lambda x: f"{x:.1%}")

#
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(df_p3.to_string(index=False))
df_p3.to_csv(OUTPUT_DIR_P3 / "phase3_summary.csv", index=False)
print(f"\nSaved → {OUTPUT_DIR_P3}/phase3_summary.csv")


# ============================================================
# visualization
# ============================================================
# ============================================================
# Cell 18: Phase 3 Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 3: Pruning Analysis', fontsize=14, fontweight='bold')

# ── 18A: （ vs ）──────────────────────
ratios_num = [0.0, 0.3, 0.5, 0.7]
acc_before_ft = [pruning_results[r]['Accuracy']   for r in ratios_num]
acc_after_ft  = [finetuned_results.get(r, pruning_results[r])['Accuracy']
                 for r in ratios_num]

x = np.arange(len(ratios_num))
w = 0.35
axes[0, 0].bar(x - w/2, acc_before_ft, w,
               label='Before Fine-tune', color='#ff7f7f', alpha=0.85)
axes[0, 0].bar(x + w/2, acc_after_ft,  w,
               label='After Fine-tune',  color='#4CAF50', alpha=0.85)
axes[0, 0].axhline(fp32_metrics['Accuracy'], color='navy',
                   linestyle='--', linewidth=1.5, label='FP32 Baseline')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([f"{int(r*100)}%" for r in ratios_num])
axes[0, 0].set_xlabel('Pruning Ratio')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy: Before vs After Fine-tuning')
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_ylim(
    min(acc_before_ft) - 0.05,
    max(acc_after_ft)  + 0.02)
axes[0, 0].grid(axis='y', alpha=0.3)

# ── 18B:  ────────────────────────────────────────
size_before = [pruning_results[r]['size_kb']   for r in ratios_num]
size_after  = [finetuned_results.get(r, pruning_results[r])['size_kb']
               for r in ratios_num]

axes[0, 1].plot(ratios_num, size_before, 'o--',
                color='#ff7f7f', label='Before FT', linewidth=2)
axes[0, 1].plot(ratios_num, size_after,  's-',
                color='#4CAF50', label='After FT',  linewidth=2)
axes[0, 1].axhline(fp32_size_kb, color='navy',
                   linestyle='--', linewidth=1.5, label='FP32 Baseline')
axes[0, 1].set_xlabel('Pruning Ratio')
axes[0, 1].set_ylabel('Model Size (KB)')
axes[0, 1].set_title('Model Size vs Pruning Ratio')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)
for r, s in zip(ratios_num, size_after):
    axes[0, 1].annotate(f"{s:.1f}KB",
                        xy=(r, s), xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', fontsize=8)

# ── 18C: Accuracy vs Size Trade-off scatter ──────────────────
all_models  = []
all_accs    = []
all_sizes   = []
all_colors  = []
color_map   = {
    'FP32'   : 'navy',
    'No'     : '#ff7f7f',
    'Yes'    : '#4CAF50',
    'Yes+QAT': '#FF9800',
}

for _, row in df_p3.iterrows():
    all_models.append(row['Model'])
    all_accs.append(row['Accuracy'])
    all_sizes.append(row['Size (KB)'])
    ft_key = row['Fine-tuned'] if row['Fine-tuned'] in color_map else 'No'
    all_colors.append(color_map.get(ft_key, 'gray'))

scatter = axes[1, 0].scatter(all_sizes, all_accs,
                              s=120, c=all_colors, zorder=5, alpha=0.9)
for i, name in enumerate(all_models):
    axes[1, 0].annotate(name,
                        xy=(all_sizes[i], all_accs[i]),
                        xytext=(5, 3), textcoords='offset points',
                        fontsize=7)
axes[1, 0].set_xlabel('Model Size (KB)')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Accuracy vs. Model Size Trade-off')
axes[1, 0].grid(alpha=0.3)

#
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='navy',    label='FP32 Baseline'),
    Patch(facecolor='#ff7f7f', label='Pruned (No FT)'),
    Patch(facecolor='#4CAF50', label='Pruned + FT'),
    Patch(facecolor='#FF9800', label='Pruned + FT + QAT'),
]
axes[1, 0].legend(handles=legend_elements, fontsize=7)

# ── 18D: （） ───────────────────────
ratios_ft   = [0.3, 0.5, 0.7]
recovery    = [
    finetuned_results[r]['Accuracy'] - pruning_results[r]['Accuracy']
    for r in ratios_ft
]
bar_colors  = ['#4CAF50' if v >= 0 else '#ff7f7f' for v in recovery]

bars = axes[1, 1].bar(
    [f"{int(r*100)}%" for r in ratios_ft],
    recovery, color=bar_colors, alpha=0.85)
axes[1, 1].axhline(0, color='black', linewidth=0.8)
axes[1, 1].set_xlabel('Pruning Ratio')
axes[1, 1].set_ylabel('Accuracy Recovery (Fine-tuning)')
axes[1, 1].set_title('Fine-tuning Recovery per Pruning Ratio')
axes[1, 1].grid(axis='y', alpha=0.3)

for bar, val in zip(bars, recovery):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{val:+.4f}",
        ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR_P3 / "phase3_analysis.png", dpi=150)
plt.show()
print(f"Saved → {OUTPUT_DIR_P3}/phase3_analysis.png")

