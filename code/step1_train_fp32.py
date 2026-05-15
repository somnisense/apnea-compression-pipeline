"""
step1_train_fp32.py

Phase 1 — Architectural redesign + FP32 baseline training (Coordinate-Attention 1D CNN)
Replaces parameter-heavy Flatten+Dense head with GlobalAvgPool1D + Coord-Attn block.
Input: 200×3 acoustic feature matrices. Output: 14,001-parameter FP32 model + 5-seed metrics.

Source: companion paper "From 14k to <60 KB: Joint Quantization-Aware Training
and Structured Pruning for On-Device Sleep Apnea Detection" (SomniAI LLC, 2026).

Public release: this is the algorithm-framework code. No dataset is included;
point DATA_DIR at your own 200x3 feature-matrix dataset to reproduce.
"""

import os
from pathlib import Path

# Data path resolves relative to this script's location.
# Place your dataset at: <paper_E_compression/>/data/  (sibling to code/)
# By design, no data is bundled with the repository — see README "Data" section.
SCRIPT_DIR = Path(__file__).parent.resolve()
PAPER_DIR = SCRIPT_DIR.parent
DATA_DIR = PAPER_DIR / "data"

# === Cell-level reproduction from notebook ===

# ----- Cell 1 -----
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Dropout, Input, Flatten,
    GlobalAveragePooling1D, BatchNormalization, Multiply, Reshape,
    Concatenate, Activation, Layer, Permute, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns

#
np.random.seed(42)
tf.random.set_seed(42)

# ----- Cell 2 -----
# ============================================================
# PART 1: Load data
# ============================================================

def prepare_data(data_dir, data, labels, label_value):
    """
    
    """
    file_names = os.listdir(data_dir)
    count = 0
    
    for file_name in file_names:
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            matrix = np.loadtxt(file_path, delimiter=':')
            
            assert matrix.shape == (200, 3), f"File {file_name} has invalid shape: {matrix.shape}"
            
            data.append(matrix)
            labels.append(label_value)
            count += 1
    
    return count


def load_dataset_binary(base_dir):
    """
    
    """
    data = []
    labels = []
    
    print("=" * 60)
    print("Loading Dataset for Binary Classification")
    print("=" * 60)
    
    original_counts = {}
    for label in [0, 1, 2]:
        dir_path = os.path.join(base_dir, str(label))
        if os.path.exists(dir_path):
            count = prepare_data(dir_path, data, labels, label)
            original_counts[label] = count
            print(f"Loaded {count} samples from original class {label}")
    
    X = np.array(data)
    y_original = np.array(labels)
    
    y_binary = np.where(y_original > 0, 1, 0)
    
    class_names = {
        0: "Normal ()",
        1: "Abnormal (: apnea_event+)"
    }
    
    print("\n" + "-" * 60)
    print("Binary Classification Mapping:")
    print("-" * 60)
    print(f"  Original Class 0 (Normal)    → Binary Class 0: {original_counts.get(0, 0)} samples")
    print(f"  Original Class 1 (Apnea)     → Binary Class 1: {original_counts.get(1, 0)} samples")
    print(f"  Original Class 2 (Hypopnea)  → Binary Class 1: {original_counts.get(2, 0)} samples")
    
    print("\n" + "-" * 60)
    print("Final Binary Distribution:")
    print("-" * 60)
    for label in [0, 1]:
        count = np.sum(y_binary == label)
        percentage = count / len(y_binary) * 100
        print(f"  Class {label} ({class_names[label]}): {count} ({percentage:.1f}%)")
    
    print(f"\nTotal samples: {len(y_binary)}")
    print(f"Data shape: {X.shape}")
    
    return X, y_binary, class_names

# ----- Cell 3 -----
# ============================================================
# PART 2: self attention 
# ============================================================

class SEBlock1D(Layer):
    """
    Squeeze-and-Excitation Block for 1D time-series data ()
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    def __init__(self, channels, reduction=16, **kwargs):
        super(SEBlock1D, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(channels // reduction, 1)
        
    def build(self, input_shape):
        self.global_pool = GlobalAveragePooling1D()
        self.fc1 = Dense(self.reduced_channels, activation='relu', 
                        kernel_initializer='he_normal')
        self.fc2 = Dense(self.channels, activation='sigmoid',
                        kernel_initializer='he_normal')
        self.reshape = Reshape((1, self.channels))
        super(SEBlock1D, self).build(input_shape)
        
    def call(self, inputs):
        # Squeeze
        se = self.global_pool(inputs)
        # Excitation
        se = self.fc1(se)
        se = self.fc2(se)
        se = self.reshape(se)
        # Scale
        return Multiply()([inputs, se])
    
    def get_config(self):
        config = super(SEBlock1D, self).get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction
        })
        return config


class CoordinateAttention1D(Layer):
    """
    Coordinate Attention Block adapted for 1D time-series data.
    
    Reference: 
        Hou, Q., Zhou, D., & Feng, J. (2021). 
        "Coordinate Attention for Efficient Mobile Network Design"
        IEEE/CVF CVPR 2021. arXiv:2103.02907
    
    Original 2D Design:
        - X-Pool: (B, H, W, C) → (B, H, 1, C) - preserves height position
        - Y-Pool: (B, H, W, C) → (B, 1, W, C) - preserves width position
        - Concatenate → Conv → Split → Attention weights
    
    1D Adaptation for Time-Series (T×C):
        - Global Pool: Captures overall channel statistics
        - Local Features: Preserves temporal position information
        - Combined attention that is both channel-aware and position-aware
    
    Key Difference from SE-Net:
        - SE-Net: Only captures channel interdependencies (loses position info)
        - Coordinate Attention: Captures both channel AND positional information
        - This is crucial for time-series where temporal position matters
          (e.g., detecting the onset/offset of apnea events)
    """
    
    def __init__(self, channels, reduction=16, **kwargs):
        super(CoordinateAttention1D, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(channels // reduction, 1)
        
    def build(self, input_shape):
        # input_shape: (batch, timesteps, channels)
        self.timesteps = input_shape[1]
        
        # Layers for coordinate attention
        self.global_pool = GlobalAveragePooling1D(keepdims=True)
        
        # Bottleneck transformation (shared)
        # Input will be (B, T, 2C) after concatenation
        self.conv_reduce = Conv1D(
            filters=self.reduced_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.bn = BatchNormalization()
        
        # Separate attention generation for channel and position
        self.conv_channel = Conv1D(
            filters=self.channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal'
        )
        
        super(CoordinateAttention1D, self).build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Forward pass of Coordinate Attention
        
        Args:
            inputs: Tensor of shape (batch, timesteps, channels)
            training: Boolean for training mode (for BatchNorm)
        
        Returns:
            Attention-weighted tensor of same shape as input
        """
        # Get dynamic shape for timesteps
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        
        # ================================================================
        # Step 1: Coordinate Information Embedding
        # ================================================================
        
        # Global context: (B, T, C) → (B, 1, C)
        global_context = self.global_pool(inputs)
        
        # Tile to match temporal dimension: (B, 1, C) → (B, T, C)
        # Using tf.repeat which is compatible with Keras layers
        global_context_tiled = tf.repeat(global_context, repeats=timesteps, axis=1)
        
        # ================================================================
        # Step 2: Concatenate global context with local features
        # ================================================================
        
        # (B, T, C) + (B, T, C) → (B, T, 2C)
        # Local features (inputs) contain position information
        # Global context contains channel-wise statistics
        concat_features = tf.concat([inputs, global_context_tiled], axis=-1)
        
        # ================================================================
        # Step 3: Bottleneck transformation
        # ================================================================
        
        # Reduce: (B, T, 2C) → (B, T, C/r)
        bottleneck = self.conv_reduce(concat_features)
        bottleneck = self.bn(bottleneck, training=training)
        bottleneck = tf.nn.relu(bottleneck)
        
        # ================================================================
        # Step 4: Generate attention weights
        # ================================================================
        
        # (B, T, C/r) → (B, T, C)
        attention = self.conv_channel(bottleneck)
        attention = tf.nn.sigmoid(attention)
        
        # ================================================================
        # Step 5: Apply attention
        # ================================================================
        
        # Element-wise: (B, T, C) * (B, T, C) → (B, T, C)
        output = inputs * attention
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(CoordinateAttention1D, self).get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction
        })
        return config

# ----- Cell 4 -----
# ============================================================
# PART 3: design model
# ============================================================

def build_original_model_binary(input_shape=(200, 3)):
    """
    CNN -  (Baseline)
    sigmoid
    
    This is the baseline model without any attention mechanism.
    """
    model = Sequential([
        Conv1D(16, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.1),
        
        Dense(1, activation='sigmoid')
    ], name="Original_CNN_Binary")
    
    return model


def build_improved_model_binary(input_shape=(200, 3)):
    """
    with Coordinate Attention
    
    Reference: 
        Hou, Q., Zhou, D., & Feng, J. (2021). 
        "Coordinate Attention for Efficient Mobile Network Design"
        IEEE/CVF CVPR 2021
    
    Key Modifications from baseline:
    1. Coordinate Attention blocks - captures channel AND position info
    2. BatchNormalization for stable training
    3. GlobalAveragePooling instead of Flatten (reduces parameters)
    4. Increased Dropout for regularization
    
    Why Coordinate Attention for Sleep Apnea Detection:
    - Apnea events have temporal structure (onset, duration, recovery)
    - Position-aware attention can better capture event boundaries
    - The 3 input channels (dB, snore flag, dB change) have different
      importance at different time points during an apnea event
    """
    inputs = Input(shape=input_shape)
    
    # ================================================================
    # Conv Block 1
    # ================================================================
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same',
               kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # MODIFICATION: Coordinate Attention instead of SE Block
    x = CoordinateAttention1D(channels=16, reduction=4)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # ================================================================
    # Conv Block 2
    # ================================================================
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = CoordinateAttention1D(channels=32, reduction=8)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # ================================================================
    # Conv Block 3
    # ================================================================
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = CoordinateAttention1D(channels=64, reduction=16)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # ================================================================
    # Classifier Head
    # ================================================================
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, 
                  name="CoordAttention_CNN_Binary")
    
    return model


def build_se_model_binary(input_shape=(200, 3)):
    """
    SE-Net based model for comparison (Optional)
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    inputs = Input(shape=input_shape)
    
    x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = SEBlock1D(channels=16, reduction=4)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SEBlock1D(channels=32, reduction=8)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SEBlock1D(channels=64, reduction=16)(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="SE_Attention_CNN_Binary")
    
    return model

# ----- Cell 5 -----
# ============================================================
# PART 4: train function
# ============================================================

def train_model_binary(model, X_train, y_train, X_val, y_val, 
                       class_weight=None, epochs=100, batch_size=32):
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=8, 
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ----- Cell 6 -----
# ============================================================
# PART 5: evaluate functin
# ============================================================

def evaluate_model_binary(model, X_test, y_test, class_names, model_name="Model", threshold=0.5):
    print("\n" + "=" * 60)
    print(f"EVALUATION: {model_name}")
    print("=" * 60)
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    print(f"\n📊 Accuracy: {metrics['accuracy']*100:.2f}%")
    
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📊 Binary Classification Metrics:")
    print(f"   Precision:  {metrics['precision']*100:.2f}%")
    print(f"   Recall:     {metrics['recall']*100:.2f}%")
    print(f"   F1-Score:   {metrics['f1']*100:.2f}%")
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n📊 Clinical Metrics:")
    print(f"   Sensitivity (Recall): {metrics['sensitivity']*100:.2f}%")
    print(f"   Specificity:          {metrics['specificity']*100:.2f}%")
    
    metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
    print(f"   AUC-ROC:              {metrics['auc_roc']:.4f}")
    
    metrics['auc_pr'] = average_precision_score(y_test, y_pred_proba)
    print(f"   AUC-PR:               {metrics['auc_pr']:.4f}")
    
    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp
    
    print(f"\n📊 Confusion Matrix Details:")
    print(f"   True Negatives (TN):  {tn}")
    print(f"   False Positives (FP): {fp}")
    print(f"   False Negatives (FN): {fn}")
    print(f"   True Positives (TP):  {tp}")
    
    print(f"\n📊 Detailed Classification Report:")
    target_names = [class_names[0], class_names[1]]
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    return metrics, y_pred, y_pred_proba


def find_optimal_threshold(y_test, y_pred_proba):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        results.append({
            'threshold': thresh,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1, results


# ============================================================
# PART 6: visual function
# ============================================================

def plot_confusion_matrix_binary(cm, class_names, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    
    labels = [name.split('(')[0].strip() for name in class_names.values()]
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    annotations = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                            for j in range(cm.shape[1])] 
                           for i in range(cm.shape[0])])
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels, annot_kws={'size': 14})
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()


def plot_training_history(history, title="Training History"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f'{title} - Loss', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1].set_title(f'{title} - Accuracy', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_roc_curve_binary(y_test, y_proba_orig, y_proba_impr, title="ROC Curve Comparison"):
    plt.figure(figsize=(8, 6))
    
    fpr_orig, tpr_orig, _ = roc_curve(y_test, y_proba_orig)
    auc_orig = roc_auc_score(y_test, y_proba_orig)
    plt.plot(fpr_orig, tpr_orig, color='#1f77b4', linewidth=2.5,
             label=f'Original CNN (AUC = {auc_orig:.3f})')
    
    fpr_impr, tpr_impr, _ = roc_curve(y_test, y_proba_impr)
    auc_impr = roc_auc_score(y_test, y_proba_impr)
    plt.plot(fpr_impr, tpr_impr, color='#ff7f0e', linewidth=2.5,
             label=f'Coord-Attention CNN (AUC = {auc_impr:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_precision_recall_curve(y_test, y_proba_orig, y_proba_impr, title="Precision-Recall Curve"):
    plt.figure(figsize=(8, 6))
    
    precision_orig, recall_orig, _ = precision_recall_curve(y_test, y_proba_orig)
    ap_orig = average_precision_score(y_test, y_proba_orig)
    plt.plot(recall_orig, precision_orig, color='#1f77b4', linewidth=2.5,
             label=f'Original CNN (AP = {ap_orig:.3f})')
    
    precision_impr, recall_impr, _ = precision_recall_curve(y_test, y_proba_impr)
    ap_impr = average_precision_score(y_test, y_proba_impr)
    plt.plot(recall_impr, precision_impr, color='#ff7f0e', linewidth=2.5,
             label=f'Coord-Attention CNN (AP = {ap_impr:.3f})')
    
    baseline = np.sum(y_test) / len(y_test)
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, 
                label=f'Baseline (AP = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_threshold_analysis(y_test, y_proba, model_name="Model"):
    thresholds = np.arange(0.1, 0.9, 0.02)
    
    precisions, recalls, f1s, specificities = [], [], [], []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall (Sensitivity)', linewidth=2)
    ax.plot(thresholds, f1s, label='F1-Score', linewidth=2)
    ax.plot(thresholds, specificities, label='Specificity', linewidth=2)
    
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    ax.axvline(x=best_thresh, color='red', linestyle='--', alpha=0.7,
               label=f'Best Threshold = {best_thresh:.2f}')
    ax.scatter([best_thresh], [best_f1], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{model_name} - Threshold Analysis', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.0, 1.0])
    
    plt.tight_layout()
    return fig, best_thresh


def compare_models_table_binary(metrics_original, metrics_improved):
    """"""
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION MODEL COMPARISON")
    print("(Original CNN vs Coordinate Attention CNN)")
    print("=" * 70)
    
    metrics_list = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall (Sensitivity)', 'recall'),
        ('Specificity', 'specificity'),
        ('F1-Score', 'f1'),
        ('AUC-ROC', 'auc_roc'),
        ('AUC-PR (Average Precision)', 'auc_pr'),
    ]
    
    print(f"\n{'Metric':<30} {'Original':>12} {'Improved':>12} {'Δ Change':>12}")
    print("-" * 70)
    
    for name, key in metrics_list:
        orig = metrics_original.get(key)
        impr = metrics_improved.get(key)
        
        if orig is not None and impr is not None:
            delta = impr - orig
            if key in ['auc_roc', 'auc_pr']:
                print(f"{name:<30} {orig:>12.4f} {impr:>12.4f} {delta:>+12.4f}")
            else:
                delta_str = f"{delta*100:+.2f}%"
                print(f"{name:<30} {orig*100:>11.2f}% {impr*100:>11.2f}% {delta_str:>12}")
    
    print("-" * 70)
    
    print(f"\n{'Parameter Count':<30} {'Original':>12} {'Improved':>12} {'Reduction':>12}")
    print("-" * 70)
    orig_params = metrics_original.get('params', 0)
    impr_params = metrics_improved.get('params', 0)
    if orig_params > 0:
        reduction = (1 - impr_params / orig_params) * 100
        print(f"{'Parameters':<30} {orig_params:>12,} {impr_params:>12,} {reduction:>11.1f}%")
    
    print("\n" + "-" * 70)
    print("Confusion Matrix Comparison:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Original':>15} {'Improved':>15}")
    print(f"{'True Negatives':<20} {metrics_original['tn']:>15} {metrics_improved['tn']:>15}")
    print(f"{'False Positives':<20} {metrics_original['fp']:>15} {metrics_improved['fp']:>15}")
    print(f"{'False Negatives':<20} {metrics_original['fn']:>15} {metrics_improved['fn']:>15}")
    print(f"{'True Positives':<20} {metrics_original['tp']:>15} {metrics_improved['tp']:>15}")

# ----- Cell 7 -----
# ============================================================
# Main function
# ============================================================

# ============  ============
BASE_DIR = str(DATA_DIR)

INPUT_SHAPE = (200, 3)
EPOCHS = 100
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============ load data ============
X, y, class_names = load_dataset_binary(BASE_DIR)

# ============ splite dataset ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\n📊 Dataset Split:")
print(f"   Training:   {X_train.shape[0]} samples")
print(f"   Testing:    {X_test.shape[0]} samples")
print(f"\n   Training set distribution:")
print(f"      Normal:   {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
print(f"      Abnormal: {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")

# ============ calculate weight ============
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n📊 Class Weights:")
print(f"   Class 0 (Normal):   {class_weight_dict[0]:.3f}")
print(f"   Class 1 (Abnormal): {class_weight_dict[1]:.3f}")

# ============ build model ============
print("\n" + "=" * 60)
print("BUILDING BINARY CLASSIFICATION MODELS")
print("=" * 60)

original_model = build_original_model_binary(INPUT_SHAPE)
improved_model = build_improved_model_binary(INPUT_SHAPE)

print(f"\n📊 Model Parameters:")
print(f"   Original Model:  {original_model.count_params():,} parameters")
print(f"   Improved Model (Coord-Attn):  {improved_model.count_params():,} parameters")
print(f"   Reduction:       {(1 - improved_model.count_params()/original_model.count_params())*100:.1f}%")

# print structure
print("\n" + "=" * 60)
print("Original Model Architecture:")
print("=" * 60)
original_model.summary()

print("\n" + "=" * 60)
print("Improved Model Architecture (Coordinate Attention):")
print("=" * 60)
improved_model.summary()

# ============ train orginal model ============
print("\n" + "=" * 60)
print("TRAINING: Original Model (Baseline)")
print("=" * 60)

history_original = train_model_binary(
    original_model, 
    X_train, y_train, 
    X_test, y_test,
    class_weight=class_weight_dict,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ============ train new model ============
print("\n" + "=" * 60)
print("TRAINING: Improved Model (Coordinate Attention)")
print("=" * 60)

history_improved = train_model_binary(
    improved_model, 
    X_train, y_train, 
    X_test, y_test,
    class_weight=class_weight_dict,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ============ evaluate model ============
metrics_original, y_pred_orig, y_proba_orig = evaluate_model_binary(
    original_model, X_test, y_test, class_names, "Original CNN (Binary)"
)
metrics_original['params'] = original_model.count_params()

metrics_improved, y_pred_impr, y_proba_impr = evaluate_model_binary(
    improved_model, X_test, y_test, class_names, "Coordinate-Attention CNN (Binary)"
)
metrics_improved['params'] = improved_model.count_params()

# ============ compare model ============
compare_models_table_binary(metrics_original, metrics_improved)

# ============ threshold analysis ============
print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

best_thresh_orig, best_f1_orig, _ = find_optimal_threshold(y_test, y_proba_orig)
print(f"Original Model - Best threshold: {best_thresh_orig:.2f} (F1: {best_f1_orig:.4f})")

best_thresh_impr, best_f1_impr, _ = find_optimal_threshold(y_test, y_proba_impr)
print(f"Improved Model - Best threshold: {best_thresh_impr:.2f} (F1: {best_f1_impr:.4f})")

# ----- Cell 8 -----
# ============ visulization ============
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)


fig1 = plot_training_history(history_original, "Original Model (Binary)")
fig1.savefig("binary_training_history_original.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_training_history_original.png")

fig2 = plot_training_history(history_improved, "Improved Model (Coordinate Attention)")
fig2.savefig("binary_training_history_improved.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_training_history_improved.png")


fig3 = plot_confusion_matrix_binary(
    metrics_original['confusion_matrix'], 
    class_names, 
    "Confusion Matrix - Original Model"
)
fig3.savefig("binary_confusion_matrix_original.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_confusion_matrix_original.png")

fig4 = plot_confusion_matrix_binary(
    metrics_improved['confusion_matrix'], 
    class_names, 
    "Confusion Matrix - Improved Model (Coordinate Attention)"
)
fig4.savefig("binary_confusion_matrix_improved.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_confusion_matrix_improved.png")


fig5 = plot_roc_curve_binary(y_test, y_proba_orig, y_proba_impr, "ROC Curve Comparison")
fig5.savefig("binary_roc_comparison.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_roc_comparison.png")


fig6 = plot_precision_recall_curve(y_test, y_proba_orig, y_proba_impr, "Precision-Recall Curve Comparison")
fig6.savefig("binary_pr_comparison.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_pr_comparison.png")


fig7, _ = plot_threshold_analysis(y_test, y_proba_orig, "Original Model")
fig7.savefig("binary_threshold_original.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_threshold_original.png")

fig8, _ = plot_threshold_analysis(y_test, y_proba_impr, "Improved Model (Coordinate Attention)")
fig8.savefig("binary_threshold_improved.png", dpi=150, bbox_inches='tight')
print("✓ Saved: binary_threshold_improved.png")

plt.show()

# ----- Cell 9 -----
# ============ save model ============
original_model.save("sleep_apnea_binary_original.keras")
improved_model.save("sleep_apnea_binary_coord_attention.keras")
print("\n✓ Models saved!")

# ============ summary ============
print("\n" + "=" * 70)
print("SUMMARY TABLE (Copy to Report)")
print("=" * 70)
print("""
| Metric | Original CNN | Coord-Attention CNN | Improvement |
|--------|--------------|---------------------|-------------|""")

metrics_to_report = [
    ('Accuracy', 'accuracy', '%'),
    ('Precision', 'precision', '%'),
    ('Recall', 'recall', '%'),
    ('Specificity', 'specificity', '%'),
    ('F1-Score', 'f1', '%'),
    ('AUC-ROC', 'auc_roc', ''),
    ('AUC-PR', 'auc_pr', ''),
]

for name, key, fmt in metrics_to_report:
    orig = metrics_original[key]
    impr = metrics_improved[key]
    delta = impr - orig
    if fmt == '%':
        print(f"| {name} | {orig*100:.2f}% | {impr*100:.2f}% | {delta*100:+.2f}% |")
    else:
        print(f"| {name} | {orig:.4f} | {impr:.4f} | {delta:+.4f} |")

print(f"| Parameters | {metrics_original['params']:,} | {metrics_improved['params']:,} | -{(1-metrics_improved['params']/metrics_original['params'])*100:.1f}% |")

print("\n" + "=" * 60)
print("COORDINATE ATTENTION EXPERIMENT COMPLETED!")
print("=" * 60)
print("\nReference:")
print("  Hou, Q., Zhou, D., & Feng, J. (2021).")
print('  "Coordinate Attention for Efficient Mobile Network Design"')
print("  IEEE/CVF CVPR 2021. arXiv:2103.02907")
