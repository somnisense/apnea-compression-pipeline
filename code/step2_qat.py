"""
step2_qat.py

Phase 2 — Quantization study (PyTorch).
Trains a PyTorch FP32 baseline of the Coord-Attn 1D architecture, then
sweeps quantization strategies: FP16, INT8 Dynamic PTQ, INT8 Static PTQ,
per-layer sensitivity, and finally Quantization-Aware Training (QAT, 10
epochs). QAT was selected as the production winner (87.58% accuracy,
58.9 KB size on this seed).

Outputs a QAT-finalized INT8 model checkpoint consumed by step3_prune.py.

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
# Phase 2: Quantization Study
# Cell 1: Environment Setup
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

# ──  ──────────────────────────────────────────────
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

# Apple M4 MPS 
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    MPS_AVAILABLE = True
    print("MPS (Apple Silicon) : Available")
else:
    MPS_AVAILABLE = False
    print("MPS (Apple Silicon) : Not available")

#  CPU （PyTorch  MPS）
DEVICE = torch.device('cpu')
print(f"Quantization device : {DEVICE}")

#
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#
OUTPUT_DIR = Path("phase2_quantization_results")
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Output dir: {OUTPUT_DIR}")


# ============================================================
# fp32_pytorch
# ============================================================
# ============================================================
# Cell 4: PyTorch Training (FP32 Baseline)
# ============================================================

def train_model(model, train_loader, val_loader,
                epochs=50, lr=1e-3, patience=10,
                save_path=None):
    """， Early Stopping。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    model.train()
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                val_correct += ((pred > 0.5).float() == yb).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc   = val_correct / len(val_loader.dataset)

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        print(f"Best model loaded from {save_path}")

    return history


# ──  FP32  ────────────────────────────────────
SAVE_PATH = str(OUTPUT_DIR / "coord_attn_fp32_best.pth")

pt_model = CoordAttnCNNBinary().to(DEVICE)
history = train_model(
    pt_model, train_loader, val_loader,
    epochs=100, lr=1e-3, patience=15,
    save_path=SAVE_PATH
)

#
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], label='Train Loss')
axes[0].plot(history['val_loss'],   label='Val Loss')
axes[0].set_title('Loss Curve'); axes[0].legend()
axes[1].plot(history['val_acc'], label='Val Accuracy', color='green')
axes[1].set_title('Validation Accuracy'); axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
plt.show()


# ============================================================
# fp32_profile
# ============================================================
# ============================================================
# Cell 6: FP32 Baseline Profiling
# ============================================================

print("Evaluating FP32 baseline model...")

fp32_metrics, fp32_probs, fp32_labels = evaluate_model(pt_model, test_loader)
fp32_latency  = measure_latency(pt_model)
fp32_size_kb  = get_model_size_kb(pt_model)

print_metrics("Coord-Attn FP32", fp32_metrics, fp32_latency, fp32_size_kb)

#
results_registry = {}
results_registry['FP32'] = {
    **fp32_metrics,
    'size_kb'    : fp32_size_kb,
    'latency_ms' : fp32_latency['mean_ms'],
    'p95_ms'     : fp32_latency['p95_ms'],
}

# ── torch.profiler  profiling ────────────────────────
dummy_input = torch.randn(1, 200, 3).to(DEVICE)
pt_model.eval()

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_flops=True
) as prof:
    with record_function("model_inference"):
        with torch.no_grad():
            for _ in range(10):
                _ = pt_model(dummy_input)

#  Top-10 
print("\n── Top-10 CPU ops by self_cpu_time_total ──")
print(prof.key_averages().table(
    sort_by="self_cpu_time_total", row_limit=10))

#  Chrome trace（ chrome://tracing ）
prof.export_chrome_trace(str(OUTPUT_DIR / "fp32_trace.json"))
print(f"\nChrome trace saved → {OUTPUT_DIR}/fp32_trace.json")


# ============================================================
# fp16_size
# ============================================================
# ============================================================
# Cell 7 (Fixed): FP16 - Model Size Measurement Only
# Apple M4 CPU  FP16 Conv1D 
# ： FP16 ， MPS （），
#        fallback  FP32（，）
# ============================================================

print("Running FP16 analysis (Apple M4 compatible)...")
print("Note: PyTorch CPU does not support FP16 Conv1D.")
print("      FP16 size is measured via state_dict; accuracy = FP32 equivalent.\n")

# ── 7A: FP16  ─────────────────────────────────
#  state_dict  FP16 ，
fp16_model_for_size = CoordAttnCNNBinary()
fp16_model_for_size.load_state_dict(
    torch.load(SAVE_PATH, map_location='cpu'))

#  state_dict  FP16
fp16_state_dict = {k: v.half() for k, v in
                   fp16_model_for_size.state_dict().items()}

fp16_save = OUTPUT_DIR / "coord_attn_fp16.pth"
torch.save(fp16_state_dict, fp16_save)
fp16_size_kb = os.path.getsize(fp16_save) / 1024
print(f"FP16 model size : {fp16_size_kb:.2f} KB  "
      f"(FP32 was {fp32_size_kb:.2f} KB, "
      f"reduction={1 - fp16_size_kb/fp32_size_kb:.1%})")

# ── 7B: （FP32 ，）──────────────────
# FP16  FP32 
fp16_eval_model = CoordAttnCNNBinary().to(DEVICE)
fp16_state_dict_fp32 = {k: v.float() for k, v in fp16_state_dict.items()}
fp16_eval_model.load_state_dict(fp16_state_dict_fp32)
fp16_eval_model.eval()

fp16_metrics, _, _ = evaluate_model(fp16_eval_model, test_loader)
print("\nFP16 weights → FP32 inference accuracy:")
for k, v in fp16_metrics.items():
    fp32_v = fp32_metrics[k]
    delta  = v - fp32_v
    print(f"  {k:<12}: {v:.4f}  (Δ vs FP32: {delta:+.4f})")

# ── 7C:  ──────────────────────────────────────────
#  MPS； CPU FP32 
if MPS_AVAILABLE:
    print("\nMeasuring FP16 latency on MPS (Apple Silicon)...")
    try:
        mps_device = torch.device('mps')

        # MPS  FP16
        fp16_mps_model = CoordAttnCNNBinary().to(mps_device)
        fp16_mps_model.load_state_dict(fp16_state_dict_fp32)
        fp16_mps_model.half()   #  FP16
        fp16_mps_model.eval()

        # Warmup
        dummy_mps = torch.randn(1, 200, 3).half().to(mps_device)
        with torch.no_grad():
            for _ in range(20):
                _ = fp16_mps_model(dummy_mps)

        #
        times = []
        with torch.no_grad():
            for _ in range(200):
                t0 = time.perf_counter()
                _ = fp16_mps_model(dummy_mps)
                torch.mps.synchronize()   #
                times.append((time.perf_counter() - t0) * 1000)

        fp16_latency = {
            'mean_ms': np.mean(times),
            'std_ms' : np.std(times),
            'p50_ms' : np.percentile(times, 50),
            'p95_ms' : np.percentile(times, 95),
        }
        print(f"  MPS FP16 Latency: {fp16_latency['mean_ms']:.3f} ± "
              f"{fp16_latency['std_ms']:.3f} ms  "
              f"(P95={fp16_latency['p95_ms']:.3f} ms)")

    except Exception as e:
        print(f"  MPS FP16 failed ({e}), falling back to CPU FP32 latency.")
        fp16_latency = measure_latency(fp16_eval_model)
        print(f"  CPU FP32 Latency (reference): "
              f"{fp16_latency['mean_ms']:.3f} ms")
else:
    print("\nMPS not available. Using CPU FP32 latency as reference.")
    fp16_latency = measure_latency(fp16_eval_model)

# ── 7D:  ─────────────────────────────────────────────
print_metrics("Coord-Attn FP16", fp16_metrics, fp16_latency, fp16_size_kb)

results_registry['FP16'] = {
    **fp16_metrics,
    'size_kb'    : fp16_size_kb,
    'latency_ms' : fp16_latency['mean_ms'],
    'p95_ms'     : fp16_latency['p95_ms'],
}

print("\n── FP16 vs FP32  ──")
print(f"  Size   : {fp32_size_kb:.2f} KB → {fp16_size_kb:.2f} KB  "
      f"({1 - fp16_size_kb/fp32_size_kb:.1%} reduction)")
print(f"  Acc    : {fp32_metrics['Accuracy']:.4f} → "
      f"{fp16_metrics['Accuracy']:.4f}  "
      f"(Δ={fp16_metrics['Accuracy']-fp32_metrics['Accuracy']:+.4f})")
print(f"  Latency: {fp32_latency['mean_ms']:.3f} ms → "
      f"{fp16_latency['mean_ms']:.3f} ms")


# ============================================================
# int8_dynamic
# ============================================================
# ============================================================
# Cell 8 (Fixed): INT8 Dynamic PTQ — Apple M4 Compatible
# ============================================================

import torch
import torch.nn as nn

# ── 8A:  ────────────────────────────────
print("Detecting available quantization engines...")

# Apple Silicon (ARM)  qnnpack；x86  fbgemm
supported_engines = torch.backends.quantized.supported_engines
print(f"Supported engines: {supported_engines}")

if 'qnnpack' in supported_engines:
    QENGINE = 'qnnpack'
elif 'fbgemm' in supported_engines:
    QENGINE = 'fbgemm'
elif 'x86' in supported_engines:
    QENGINE = 'x86'
else:
    QENGINE = 'none'
    print("⚠️  WARNING: No quantization engine found!")

torch.backends.quantized.engine = QENGINE
print(f"✅ Using quantization engine: {QENGINE}")

# ── 8B:  FP32  ───────────────────────────────────
int8_dyn_model = CoordAttnCNNBinary().to(DEVICE)
int8_dyn_model.load_state_dict(
    torch.load(SAVE_PATH, map_location=DEVICE))
int8_dyn_model.eval()

# ── 8C:  Linear  ────────────────────────────────
print("\nLinear layers found in model:")
for name, module in int8_dyn_model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"  [{name}]: in={module.in_features}, "
              f"out={module.out_features}")

# ── 8D:  INT8 Dynamic  ───────────────────────────
try:
    int8_dyn_model = torch.quantization.quantize_dynamic(
        int8_dyn_model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )
    print("\n✅ INT8 Dynamic quantization successful!")
    print("\nQuantized model structure:")
    print(int8_dyn_model)

except RuntimeError as e:
    print(f"\n❌ quantize_dynamic failed: {e}")
    print("Falling back to manual per-layer quantization...")

    # ── Fallback:  Linear  ──────────────
    def manual_dynamic_quantize(model, engine='qnnpack'):
        """
         nn.Linear 。
         quantize_dynamic 。
        """
        torch.backends.quantized.engine = engine

        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear):
                #
                weight = module.weight.data
                bias   = module.bias.data if module.bias is not None else None

                #
                q_weight = torch.quantize_per_tensor(
                    weight,
                    scale=weight.abs().max() / 127.0,
                    zero_point=0,
                    dtype=torch.qint8
                )

                #  Linear
                q_linear = torch.ao.nn.quantized.dynamic.Linear(
                    module.in_features,
                    module.out_features,
                    bias_=module.bias is not None,
                    dtype=torch.qint8
                )
                q_linear.set_weight_bias(q_weight, bias)
                setattr(model, name, q_linear)
                print(f"  ✅ Replaced [{name}] with DynamicQuantizedLinear")

            else:
                #
                manual_dynamic_quantize(module, engine)

        return model

    int8_dyn_model = manual_dynamic_quantize(int8_dyn_model, engine=QENGINE)
    print("\n✅ Manual dynamic quantization complete!")

# ── 8E:  ─────────────────────────────────────────────
print("\nEvaluating INT8 Dynamic model...")
int8_dyn_metrics, _, _ = evaluate_model(int8_dyn_model, test_loader)
int8_dyn_latency        = measure_latency(int8_dyn_model)

#
int8_dyn_save = OUTPUT_DIR / "coord_attn_int8_dynamic.pth"
torch.save(int8_dyn_model.state_dict(), int8_dyn_save)
int8_dyn_size_kb = get_model_size_kb(int8_dyn_save)

print_metrics("Coord-Attn INT8-Dynamic",
              int8_dyn_metrics, int8_dyn_latency, int8_dyn_size_kb)

# ── 8F:  FP32  ─────────────────────────────────────
print("\n── INT8-Dynamic vs FP32  ──")
for metric in ['Accuracy', 'F1', 'AUC-ROC']:
    fp32_v   = fp32_metrics[metric]
    int8_v   = int8_dyn_metrics[metric]
    delta    = int8_v - fp32_v
    status   = "✅" if abs(delta) < 0.01 else "⚠️"
    print(f"  {status} {metric:<12}: FP32={fp32_v:.4f} → "
          f"INT8={int8_v:.4f}  (Δ={delta:+.4f})")

print(f"\n  📦 Size   : {fp32_size_kb:.2f} KB → {int8_dyn_size_kb:.2f} KB  "
      f"({1 - int8_dyn_size_kb/fp32_size_kb:.1%} reduction)")
print(f"  ⚡ Latency : {fp32_latency['mean_ms']:.3f} ms → "
      f"{int8_dyn_latency['mean_ms']:.3f} ms")

# ── 8G:  ─────────────────────────────────────────
results_registry['INT8-Dynamic'] = {
    **int8_dyn_metrics,
    'size_kb'    : int8_dyn_size_kb,
    'latency_ms' : int8_dyn_latency['mean_ms'],
    'p95_ms'     : int8_dyn_latency['p95_ms'],
}

print("\n✅ Cell 8 ")


# ============================================================
# int8_static
# ============================================================
# ============================================================
# Cell 9 (Fixed): INT8 Static PTQ — Linear-Only Quantization
# Apple M4 / qnnpack 
# ： fc1 / fc2  Linear 
#       Conv1D / BN / Attention  FP32
# ============================================================

import torch
import torch.nn as nn
import torch.quantization as tq
from copy import deepcopy

torch.backends.quantized.engine = QENGINE
print(f"Quantization engine : {QENGINE}")
print("Running INT8 Static PTQ (Linear-only)...\n")

# ── 9A:  Linear-only  ──────────────────────────
class LinearOnlyQuantizableCNN(CoordAttnCNNBinary):
    """
     fc1 → fc2  QuantStub/DeQuantStub。
    Conv1D / BN / Attention  FP32。
    """
    def __init__(self):
        super().__init__()
        self.quant   = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        # ── FP32 ：Conv + Attention ──────────────────────
        x = x.permute(0, 2, 1)                              # (B,3,200)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.attn1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.attn2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.attn3(x)

        x = self.gap(x).squeeze(-1)                         # (B,64) FP32

        # ── INT8 ：fc1 + fc2 ─────────────────────────
        x = self.quant(x)                                   # FP32 → INT8
        x = self.relu_fc(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.dequant(x)                                 # INT8 → FP32

        return torch.sigmoid(x)


# ── 9B:  ────────────────────────────────────
int8_static_model = LinearOnlyQuantizableCNN().to(DEVICE)

#  FP32 （strict=False  quant/dequant stub）
state_dict = torch.load(SAVE_PATH, map_location=DEVICE)
missing, unexpected = int8_static_model.load_state_dict(
    state_dict, strict=False)
print(f"Missing keys    : {missing}")      # : quant/dequant  stub
print(f"Unexpected keys : {unexpected}")   # : []
int8_static_model.eval()

# ── 9C:  qconfig ──────────────────────────────────────────
#  None（）
int8_static_model.qconfig = None

#  fc1 / fc2  qconfig
qconfig = tq.get_default_qconfig(QENGINE)
int8_static_model.fc1.qconfig    = qconfig
int8_static_model.fc2.qconfig    = qconfig
int8_static_model.quant.qconfig  = qconfig
int8_static_model.dequant.qconfig = qconfig

# （ qconfig ）
tq.prepare(int8_static_model, inplace=True)
print("Quantization prepare done (Linear layers only).")

# ── 9D: （ 200 ）─────────────────────────────
print("Calibrating with 200 samples...")
int8_static_model.eval()
with torch.no_grad():
    for xb, _ in calib_loader:
        int8_static_model(xb.to(DEVICE))
print("Calibration done.")

# ── 9E:  INT8  ────────────────────────────────
tq.convert(int8_static_model, inplace=True)
print("INT8 Static conversion done.\n")

# ── 9F:  ──────────────────────────────────────────
print("Quantized layers in model:")
for name, module in int8_static_model.named_modules():
    module_type = type(module).__name__
    if 'Quantized' in module_type or 'Quant' in module_type:
        print(f"  ✅ [{name}] → {module_type}")

# ── 9G:  ─────────────────────────────────────────────────
print("\nEvaluating INT8 Static model...")
int8_static_metrics, _, _ = evaluate_model(int8_static_model, test_loader)
int8_static_latency        = measure_latency(int8_static_model)

#
int8_static_save = OUTPUT_DIR / "coord_attn_int8_static.pth"
torch.save(int8_static_model.state_dict(), int8_static_save)
int8_static_size_kb = get_model_size_kb(int8_static_save)

print_metrics("Coord-Attn INT8-Static",
              int8_static_metrics, int8_static_latency, int8_static_size_kb)

# ── 9H:  FP32  ─────────────────────────────────────────
print("\n── INT8-Static vs FP32  ──")
for metric in ['Accuracy', 'F1', 'AUC-ROC']:
    fp32_v = fp32_metrics[metric]
    int8_v = int8_static_metrics[metric]
    delta  = int8_v - fp32_v
    status = "✅" if abs(delta) < 0.01 else "⚠️"
    print(f"  {status} {metric:<12}: "
          f"FP32={fp32_v:.4f} → INT8={int8_v:.4f}  (Δ={delta:+.4f})")

print(f"\n  📦 Size   : {fp32_size_kb:.2f} KB → "
      f"{int8_static_size_kb:.2f} KB  "
      f"({1 - int8_static_size_kb/fp32_size_kb:.1%} reduction)")
print(f"  ⚡ Latency : {fp32_latency['mean_ms']:.3f} ms → "
      f"{int8_static_latency['mean_ms']:.3f} ms")

# ── 9I:  ─────────────────────────────────────────────
results_registry['INT8-Static'] = {
    **int8_static_metrics,
    'size_kb'    : int8_static_size_kb,
    'latency_ms' : int8_static_latency['mean_ms'],
    'p95_ms'     : int8_static_latency['p95_ms'],
}

print("\n✅ Cell 9 ")


# ============================================================
# per_layer
# ============================================================
# ============================================================
# Cell 10: Per-Layer Quantization Sensitivity Analysis
# ============================================================

print("Running per-layer sensitivity analysis...")

#  Dynamic INT8，
layer_names = ['fc1', 'fc2']   # Linear （Conv1D dynamic ）

sensitivity_results = {}

for target_layer in layer_names:
    model_tmp = CoordAttnCNNBinary().to(DEVICE)
    model_tmp.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model_tmp.eval()

    #  target_layer
    target_module = {target_layer: getattr(model_tmp, target_layer)}
    model_tmp = torch.quantization.quantize_dynamic(
        model_tmp,
        qconfig_spec=set([type(getattr(model_tmp, target_layer))]),
        dtype=torch.qint8
    )

    metrics, _, _ = evaluate_model(model_tmp, test_loader)
    acc_drop = fp32_metrics['Accuracy'] - metrics['Accuracy']
    f1_drop  = fp32_metrics['F1']       - metrics['F1']

    sensitivity_results[target_layer] = {
        'Accuracy'    : metrics['Accuracy'],
        'Acc_Drop'    : acc_drop,
        'F1'          : metrics['F1'],
        'F1_Drop'     : f1_drop,
    }
    print(f"Layer [{target_layer}] → "
          f"Acc={metrics['Accuracy']:.4f} (Δ={acc_drop:+.4f}), "
          f"F1={metrics['F1']:.4f} (Δ={f1_drop:+.4f})")

# ──  ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

layers = list(sensitivity_results.keys())
acc_drops = [sensitivity_results[l]['Acc_Drop'] for l in layers]
f1_drops  = [sensitivity_results[l]['F1_Drop']  for l in layers]

# Accuracy drop bar chart
colors = ['red' if d > 0.005 else 'orange' if d > 0.001 else 'green'
          for d in acc_drops]
axes[0].barh(layers, acc_drops, color=colors)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title('Accuracy Drop per Layer (INT8 Dynamic)')
axes[0].set_xlabel('Accuracy Drop (↑ worse)')

# F1 drop bar chart
colors2 = ['red' if d > 0.005 else 'orange' if d > 0.001 else 'green'
           for d in f1_drops]
axes[1].barh(layers, f1_drops, color=colors2)
axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title('F1-Score Drop per Layer (INT8 Dynamic)')
axes[1].set_xlabel('F1 Drop (↑ worse)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensitivity_analysis.png", dpi=150)
plt.show()

print(f"\nSensitivity analysis saved → {OUTPUT_DIR}/sensitivity_analysis.png")


# ============================================================
# qat
# ============================================================
# ============================================================
# Cell 11 (Fixed v2): QAT —  Conv1D 
# Apple M4 / qnnpack 
# ============================================================

import torch
import torch.nn as nn
import torch.quantization as tq
from copy import deepcopy

torch.backends.quantized.engine = QENGINE
print(f"Quantization engine : {QENGINE}")
print("Starting QAT (Linear-only, Conv1D excluded)...\n")

# ── ： QAT  ──────────────────────────
#  qconfig ， nn.Module 
#  prepare_qat  Conv 

class QATLinearOnlyCNN(nn.Module):
    """
    QAT 。
     CoordAttnCNNBinary ，：
    1.  QuantizableCoordAttnCNN（ qconfig ）
    2. QuantStub/DeQuantStub  fc1→fc2
    3. Conv1D / BN / Attention 
    """
    def __init__(self):
        super().__init__()

        # ── Conv blocks（ FP32）────────────────────────
        self.conv1 = nn.Conv1d(3,  16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(16)
        self.attn1 = CoordinateAttention1D(16, reduction=4)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(32)
        self.attn2 = CoordinateAttention1D(32, reduction=8)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(64)
        self.attn3 = CoordinateAttention1D(64, reduction=16)
        self.pool3 = nn.MaxPool1d(2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        # ── Quantization boundary ──────────────────────────
        self.quant   = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # ── Linear classifier（）─────────────────
        self.fc1     = nn.Linear(64, 64)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop    = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, 1)

    def forward(self, x):
        # ── FP32  ──────────────────────────────────────
        x = x.permute(0, 2, 1)                    # (B,3,200)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.attn1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.attn2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.attn3(x)

        x = self.gap(x).squeeze(-1)               # (B,64) FP32

        # ── INT8  ──────────────────────────────────
        x = self.quant(x)                          # FP32→INT8
        x = self.relu_fc(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.dequant(x)                        # INT8→FP32

        return torch.sigmoid(x)

    def load_from_fp32(self, fp32_state_dict):
        """ FP32 ， quant/dequant stub。"""
        own_state = self.state_dict()
        for name, param in fp32_state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
        print("  FP32 weights loaded into QAT model.")


def set_qat_qconfig_linear_only(model, engine):
    """
     qconfig ：
    -  None（）
    -  quant / dequant / fc1 / fc2  QAT qconfig
    - Conv1D / BN / Attention  None →  prepare_qat 
    """
    qat_cfg = tq.get_default_qat_qconfig(engine)

    # Step 1: 
    for module in model.modules():
        module.qconfig = None

    # Step 2: 
    model.quant.qconfig   = qat_cfg
    model.dequant.qconfig = qat_cfg
    model.fc1.qconfig     = qat_cfg
    model.fc2.qconfig     = qat_cfg

    # Step 3: 
    print("  qconfig assignment:")
    for name, mod in model.named_modules():
        cfg = getattr(mod, 'qconfig', None)
        if cfg is not None:
            print(f"    ✅ [{name}] → QAT qconfig")
        elif isinstance(mod, (nn.Conv1d, nn.BatchNorm1d)):
            print(f"    ⛔ [{name}] → None (excluded)")


def run_qat(epochs, base_weights_path,
            train_loader, val_loader, test_loader):

    print(f"\n{'─'*45}")
    print(f"  QAT Training — {epochs} epochs")
    print(f"{'─'*45}")

    # ── 1.  ───────────────────────────────────────
    model = QATLinearOnlyCNN().to(DEVICE)

    fp32_state = torch.load(base_weights_path, map_location=DEVICE)
    model.load_from_fp32(fp32_state)

    # ── 2.  qconfig（ Linear ）──────────────────
    set_qat_qconfig_linear_only(model, QENGINE)

    # ── 3. prepare_qat（ qconfig ）────────────
    tq.prepare_qat(model, inplace=True)

    # ── 4. ： Conv1D  ────────────────────
    print("\n  Post-prepare verification:")
    for name, mod in model.named_modules():
        mod_type = type(mod).__name__
        if 'FakeQuantize' in mod_type or 'Observer' in mod_type:
            print(f"    🔵 FakeQuant active : [{name}]")
        if isinstance(mod, nn.Conv1d):
            has_fq = any('FakeQuantize' in type(c).__name__
                         for c in mod.children())
            status = "⚠️  HAS FakeQuant" if has_fq else "✅ FP32 (clean)"
            print(f"    {status} : [{name}] Conv1d")

    # ── 5. QAT  ─────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_acc = 0.0
    model.train()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #  10 epoch  epoch 
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb.to(DEVICE))
                    correct += ((pred > 0.5).float() ==
                                yb.to(DEVICE)).sum().item()
            val_acc = correct / len(val_loader.dataset)
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch:>3d}/{epochs} | "
                  f"loss={avg_loss:.4f} | val_acc={val_acc:.4f}")
            best_val_acc = max(best_val_acc, val_acc)

    # ── 6.  INT8  ───────────────────────────
    model.eval()
    tq.convert(model, inplace=True)
    print(f"\n  ✅ QAT convert done. Best val_acc={best_val_acc:.4f}")

    # ── 7.  ────────────────────────────────────
    print("  Post-convert structure check:")
    for name, mod in model.named_modules():
        mod_type = type(mod).__name__
        if 'Quantized' in mod_type or 'quantized' in mod_type.lower():
            print(f"    ✅ INT8 layer : [{name}] → {mod_type}")
        if isinstance(mod, nn.Conv1d):
            print(f"    ✅ FP32 kept  : [{name}] → Conv1d")

    # ── 8.  ────────────────────────────────────────
    metrics, _, _ = evaluate_model(model, test_loader)
    latency       = measure_latency(model)

    save_path = OUTPUT_DIR / f"coord_attn_qat_{epochs}ep.pth"
    torch.save(model.state_dict(), save_path)
    size_kb = get_model_size_kb(save_path)

    return metrics, latency, size_kb


# ──  QAT  ─────────────────────────────────────
QAT_EPOCHS_LIST = [10, 20, 50]
qat_results     = {}

for ep in QAT_EPOCHS_LIST:
    m, lat, sz = run_qat(
        ep, SAVE_PATH,
        train_loader, val_loader, test_loader)

    key = f'QAT-{ep}ep'
    qat_results[key] = {
        **m,
        'size_kb'    : sz,
        'latency_ms' : lat['mean_ms'],
        'p95_ms'     : lat['p95_ms'],
    }
    print_metrics(f"QAT {ep} epochs", m, lat, sz)
    results_registry[key] = qat_results[key]

print("\n✅ Cell 11 ")


# ============================================================
# summary
# ============================================================
# ============================================================
# Cell 12: Phase 2 Summary & Visualization
# ============================================================

# ── 12A:  ─────────────────────────────────────────
summary_df = pd.DataFrame(results_registry).T
summary_df = summary_df[['Accuracy','F1','AUC-ROC','size_kb',
                          'latency_ms','p95_ms']]
summary_df.index.name = 'Model'
summary_df = summary_df.round(4)

print("\n" + "="*70)
print("PHASE 2 QUANTIZATION SUMMARY")
print("="*70)
print(summary_df.to_string())
summary_df.to_csv(OUTPUT_DIR / "phase2_summary.csv")
print(f"\nSaved → {OUTPUT_DIR}/phase2_summary.csv")

# ── 12B: Trade-off （ vs ）────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models   = summary_df.index.tolist()
accs     = summary_df['Accuracy'].values
sizes    = summary_df['size_kb'].values
latencies= summary_df['latency_ms'].values
colors   = plt.cm.Set1(np.linspace(0, 1, len(models)))

# Plot 1: Accuracy vs Model Size
for i, (m, acc, sz) in enumerate(zip(models, accs, sizes)):
    axes[0].scatter(sz, acc, s=120, color=colors[i], label=m, zorder=5)
axes[0].set_xlabel('Model Size (KB)')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy vs. Model Size Trade-off')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy vs Latency
for i, (m, acc, lat) in enumerate(zip(models, accs, latencies)):
    axes[1].scatter(lat, acc, s=120, color=colors[i], label=m, zorder=5)
axes[1].set_xlabel('Inference Latency (ms)')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs. Latency Trade-off')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "phase2_tradeoff_curves.png", dpi=150)
plt.show()

# ── 12C:  FP32  bar chart ─────────────
fp32_acc  = results_registry['FP32']['Accuracy']
fp32_size = results_registry['FP32']['size_kb']

acc_changes  = [(results_registry[m]['Accuracy'] - fp32_acc) * 100
                for m in models]
size_changes = [(1 - results_registry[m]['size_kb'] / fp32_size) * 100
                for m in models]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bar_colors_acc  = ['green' if v >= 0 else 'red' for v in acc_changes]
bar_colors_size = ['steelblue'] * len(models)

axes[0].bar(models, acc_changes, color=bar_colors_acc)
axes[0].axhline(0, color='black', lw=0.8)
axes[0].set_title('Accuracy Change vs. FP32 (%)')
axes[0].set_ylabel('Δ Accuracy (%)')
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(models, size_changes, color=bar_colors_size)
axes[1].set_title('Model Size Reduction vs. FP32 (%)')
axes[1].set_ylabel('Size Reduction (%)')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "phase2_delta_charts.png", dpi=150)
plt.show()

print("\nPhase 2 ！:", OUTPUT_DIR)

