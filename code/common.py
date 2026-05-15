"""
common.py

Shared PyTorch utilities for the compression pipeline (this paper):
the Coord-Attn 1D model definition, the 200x3 feature-matrix data loader,
and the per-event evaluation metric helpers used across steps 2-4.

Place your dataset at <paper_E_compression>/data/ (sibling to code/).
By design, no data is bundled with the repository.

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


# ============================================================
# model_def
# ============================================================
# ============================================================
# Cell 2: PyTorch Model Definition
# ============================================================

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation Block for 1D sequences."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        b, c, _ = x.shape
        s = self.gap(x).view(b, c)
        s = self.fc(s).view(b, c, 1)
        return x * s


class CoordinateAttention1D(nn.Module):
    """
    1D Coordinate Attention Block.
    Reference: Hou et al., CVPR 2021 (adapted for 1D).
    
    Input : (B, C, L)  — PyTorch channel-first format
    Output: (B, C, L)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(1, channels // reduction)

        # Channel + Position encoding
        self.pool = nn.AdaptiveAvgPool1d(1)          # global context
        self.pos_enc = nn.AdaptiveAvgPool1d(None)    # position-aware (identity)

        self.conv_h = nn.Conv1d(channels, mid, kernel_size=1, bias=False)
        self.bn_h   = nn.BatchNorm1d(mid)
        self.act    = nn.ReLU(inplace=True)

        self.conv_ch = nn.Conv1d(mid, channels, kernel_size=1, bias=False)
        self.conv_cw = nn.Conv1d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, L)
        b, c, l = x.shape

        # --- Channel attention (global) ---
        ch_attn = self.pool(x)                        # (B, C, 1)

        # --- Position attention (local) ---
        # Transpose → pool along channel dim → transpose back
        x_t = x.permute(0, 2, 1)                     # (B, L, C)
        pos_attn = x_t.mean(dim=2, keepdim=True)      # (B, L, 1)
        pos_attn = pos_attn.permute(0, 2, 1)          # (B, 1, L)
        pos_attn = pos_attn.expand(b, c, l)           # (B, C, L)

        # --- Combine & encode ---
        # Concatenate along length dim: (B, C, 1+L)
        combined = torch.cat([ch_attn, pos_attn], dim=2)  # (B, C, 1+L)
        # Reduce channels
        combined = self.act(self.bn_h(self.conv_h(combined)))  # (B, mid, 1+L)

        # Split back
        ch_part  = combined[:, :, :1]                 # (B, mid, 1)
        pos_part = combined[:, :, 1:]                 # (B, mid, L)

        # Generate attention maps
        ch_attn_map  = torch.sigmoid(self.conv_ch(ch_part))   # (B, C, 1)
        pos_attn_map = torch.sigmoid(self.conv_cw(pos_part))  # (B, C, L)

        return x * ch_attn_map * pos_attn_map


class CoordAttnCNNBinary(nn.Module):
    """
    PyTorch equivalent of build_improved_model_binary (Keras).
    Input shape: (B, 200, 3)  → permute to (B, 3, 200) for Conv1D
    Output: (B, 1) sigmoid probability
    """
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv1d(3,  16, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(16)
        self.attn1 = CoordinateAttention1D(16, reduction=4)
        self.pool1 = nn.MaxPool1d(2)

        # Block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(32)
        self.attn2 = CoordinateAttention1D(32, reduction=8)
        self.pool2 = nn.MaxPool1d(2)

        # Block 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(64)
        self.attn3 = CoordinateAttention1D(64, reduction=16)
        self.pool3 = nn.MaxPool1d(2)

        # Classifier
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(64, 64)
        self.relu_fc = nn.ReLU(inplace=True)
        self.drop    = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 200, 3) → (B, 3, 200)
        x = x.permute(0, 2, 1)

        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.attn1(x)

        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.attn2(x)

        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.attn3(x)

        x = self.gap(x).squeeze(-1)          # (B, 64)
        x = self.drop(self.relu_fc(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))       # (B, 1)
        return x


# ──  ──────────────────────────────────────────────
model_test = CoordAttnCNNBinary()
dummy = torch.randn(4, 200, 3)
out   = model_test(dummy)
print(f"Output shape : {out.shape}")          # (4, 1)
print(f"Total params : {sum(p.numel() for p in model_test.parameters()):,}")


# ============================================================
# data_loader
# ============================================================
# ============================================================
# Phase 2: Cell 3 - Data Loading & Dataset Preparation
# ============================================================

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ── （）────────────────────────────

def prepare_data(data_dir, data, labels, label_value):
    """
     txt 。
     (200, 3) ， ':'。
    """
    file_names = os.listdir(data_dir)
    count = 0

    for file_name in sorted(file_names):        # sorted 
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            matrix = np.loadtxt(file_path, delimiter=':')

            assert matrix.shape == (200, 3), \
                f"File {file_name} has invalid shape: {matrix.shape}"

            data.append(matrix)
            labels.append(label_value)
            count += 1

    return count


def load_dataset_binary(base_dir):
    """
    ：
         Class 0 (Normal)   → Binary 0
         Class 1 (Apnea)    → Binary 1
         Class 2 (Hypopnea) → Binary 1
    """
    data, labels = [], []

    print("=" * 60)
    print("Loading Dataset for Binary Classification")
    print("=" * 60)

    original_counts = {}
    for label in [0, 1, 2]:
        dir_path = os.path.join(base_dir, str(label))
        if os.path.exists(dir_path):
            count = prepare_data(dir_path, data, labels, label)
            original_counts[label] = count
            print(f"  Loaded {count:>4d} samples from class {label}  →  {dir_path}")
        else:
            print(f"  [WARN] Directory not found, skipped: {dir_path}")
            original_counts[label] = 0

    X          = np.array(data,   dtype=np.float32)   # (N, 200, 3)
    y_original = np.array(labels, dtype=np.int64)

    #  → 
    y_binary = np.where(y_original > 0, 1, 0).astype(np.float32)

    class_names = {
        0: "Normal ()",
        1: "Abnormal (:  + )"
    }

    print("\n" + "-" * 60)
    print("Binary Classification Mapping:")
    print("-" * 60)
    print(f"  Class 0 (Normal)    → Binary 0 : {original_counts.get(0, 0):>4d} samples")
    print(f"  Class 1 (Apnea)     → Binary 1 : {original_counts.get(1, 0):>4d} samples")
    print(f"  Class 2 (Hypopnea)  → Binary 1 : {original_counts.get(2, 0):>4d} samples")

    print("\n" + "-" * 60)
    print("Final Binary Distribution:")
    print("-" * 60)
    for lbl in [0, 1]:
        cnt = int(np.sum(y_binary == lbl))
        pct = cnt / len(y_binary) * 100
        print(f"  Class {lbl} ({class_names[lbl]}): {cnt:>4d}  ({pct:.1f}%)")

    print(f"\n  Total samples : {len(y_binary)}")
    print(f"  Data shape    : {X.shape}")          # (N, 200, 3)
    print("=" * 60)

    return X, y_binary, class_names


# ── 3A:  ──────────────────────────────────────────

BASE_DIR = str(DATA_DIR)

X_all, y_all, CLASS_NAMES = load_dataset_binary(BASE_DIR)

# ── 3B: Train / Val / Test  ───────────────────────────────
#   ：70% train / 15% val / 15% test（stratify ）

X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all,
    test_size=0.15,
    random_state=SEED,
    stratify=y_all
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.15 / 0.85,     #  15%
    random_state=SEED,
    stratify=y_temp
)

print(f"Train : {X_train.shape}  |  pos ratio = {y_train.mean():.3f}")
print(f"Val   : {X_val.shape}    |  pos ratio = {y_val.mean():.3f}")
print(f"Test  : {X_test.shape}   |  pos ratio = {y_test.mean():.3f}")

# ── 3C:  PyTorch DataLoader ───────────────────────────────

def make_loader(X: np.ndarray, y: np.ndarray,
                batch_size: int = 64,
                shuffle: bool = False) -> DataLoader:
    """
     numpy  PyTorch DataLoader。
    X shape : (N, 200, 3)  →  Tensor float32
    y shape : (N,)         →  Tensor float32，unsqueeze → (N, 1)
    """
    X_t = torch.from_numpy(X)                          # (N, 200, 3)
    y_t = torch.from_numpy(y).unsqueeze(1)             # (N, 1)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


BATCH_SIZE = 64

train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

# ： INT8 Static PTQ， 200 
calib_loader = make_loader(X_train[:200], y_train[:200],
                           batch_size=200, shuffle=False)

# ── 3D:  ────────────────────────────────────────

print("\n── DataLoader Sanity Check ──")
xb, yb = next(iter(train_loader))
print(f"  Batch X shape : {xb.shape}   dtype={xb.dtype}")   # (64, 200, 3)
print(f"  Batch y shape : {yb.shape}   dtype={yb.dtype}")   # (64, 1)
print(f"  X value range : [{xb.min():.3f}, {xb.max():.3f}]")
print(f"  y unique vals : {yb.unique().tolist()}")
print("  ✅ Data loading complete.")


# ============================================================
# eval_utils
# ============================================================
# ============================================================
# Cell 5: Evaluation Utilities
# ============================================================

def evaluate_model(model, loader, threshold=0.5, device=DEVICE):
    """。"""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            prob = model(xb).cpu().numpy().flatten()
            all_probs.extend(prob)
            all_labels.extend(yb.numpy().flatten())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs > threshold).astype(int)

    metrics = {
        'Accuracy'   : accuracy_score(labels, preds),
        'Precision'  : precision_score(labels, preds, zero_division=0),
        'Recall'     : recall_score(labels, preds, zero_division=0),
        'F1'         : f1_score(labels, preds, zero_division=0),
        'AUC-ROC'    : roc_auc_score(labels, probs),
        'AUC-PR'     : average_precision_score(labels, probs),
        'Specificity': recall_score(labels, preds, pos_label=0, zero_division=0),
    }
    return metrics, probs, labels


def measure_latency(model, input_shape=(1, 200, 3),
                    n_warmup=20, n_runs=200, device=DEVICE):
    """（ms）。"""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        'mean_ms' : np.mean(times),
        'std_ms'  : np.std(times),
        'p50_ms'  : np.percentile(times, 50),
        'p95_ms'  : np.percentile(times, 95),
        'p99_ms'  : np.percentile(times, 99),
    }


def get_model_size_kb(model_or_path):
    """（KB）。"""
    if isinstance(model_or_path, (str, Path)):
        return os.path.getsize(model_or_path) / 1024
    #
    tmp = OUTPUT_DIR / "_tmp_size_check.pth"
    torch.save(model_or_path.state_dict(), tmp)
    size = os.path.getsize(tmp) / 1024
    os.remove(tmp)
    return size


def print_metrics(name, metrics, latency=None, size_kb=None):
    print(f"\n{'='*55}")
    print(f"  Model: {name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    if latency:
        print(f"  Latency     : {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")
        print(f"  P95 Latency : {latency['p95_ms']:.3f} ms")
    if size_kb:
        print(f"  Model Size  : {size_kb:.2f} KB")
    print(f"{'='*55}")

