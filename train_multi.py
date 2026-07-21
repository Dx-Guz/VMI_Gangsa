# train_multi.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Pastikan model.py sudah di-update ke versi multi-output
from model_multi import build_model

# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "models/gangsa_bilstm_multi.h5"
LABEL_MAP_PATH = "models/label_map_multi.npy"
HISTORY_PLOT_PATH = "models/training_history_multi.png"
CONFIG_SAVE_PATH = "models/training_config_multi.json"

ACTIONS = ['pukul', 'redam', 'netral']
SEQUENCE_LENGTH = 15
FEATURE_DIM = 128
NUM_CLASSES = len(ACTIONS)

EPOCHS = 50
BATCH_SIZE = 16
TEST_RATIO = 0.2
VAL_RATIO = 0.2
RANDOM_SEED = 42

# =========================
# LOAD DATASET (1 LABEL → 2 LABEL)
# =========================
def load_dataset_multi():
    data, labels_r, labels_l = [], [], []
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder '{DATASET_PATH}' tidak ditemukan!")

    for action in ACTIONS:
        folder = os.path.join(DATASET_PATH, action)
        if not os.path.exists(folder):
            print(f"⚠️ Folder {folder} tidak ada, dilewati.")
            continue

        for file in os.listdir(folder):
            if file.endswith(".npy"):
                path = os.path.join(folder, file)
                try:
                    seq = np.load(path)
                    if seq.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                        data.append(seq)
                        # Pemetaan deterministik berdasarkan nama folder
                        if action == 'pukul':
                            labels_r.append(0)  # pukul
                            labels_l.append(2)  # netral
                        elif action == 'redam':
                            labels_r.append(2)  # netral
                            labels_l.append(1)  # redam
                        else:  # netral
                            labels_r.append(2)
                            labels_l.append(2)
                    else:
                        print(f"⚠️ Skip {file}: shape {seq.shape} != ({SEQUENCE_LENGTH}, {FEATURE_DIM})")
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")

    if len(data) == 0:
        raise ValueError("Tidak ada data valid yang ditemukan di dataset!")

    X = np.array(data)
    y_right = to_categorical(labels_r, num_classes=NUM_CLASSES)
    y_left = to_categorical(labels_l, num_classes=NUM_CLASSES)
    
    print(f"📦 Loaded {len(X)} sequences | X: {X.shape} | y_right: {y_right.shape} | y_left: {y_left.shape}")
    return X, y_right, y_left

# =========================
# VISUALIZATION HELPER
# =========================
def plot_training_history(history, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve (Total Loss)
    ax1.plot(history.history['loss'], label='Total Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Total Val Loss', linewidth=2)
    ax1.set_title('Total Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy Curve (Fallback ke key yang tersedia di Keras)
    acc_key = 'accuracy' if 'accuracy' in history.history else 'right_output_accuracy'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_right_output_accuracy'
    
    if acc_key in history.history:
        ax2.plot(history.history[acc_key], label='Train Acc', linewidth=2)
        ax2.plot(history.history[val_acc_key], label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Training history plot saved to {save_path}")

# =========================
# MAIN TRAINING
# =========================
def main():
    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    X, y_right, y_left = load_dataset_multi()

    # 2. Consistent 3-Way Split
    X_temp, X_test, yr_temp, yr_test, yl_temp, yl_test = train_test_split(
        X, y_right, y_left, test_size=TEST_RATIO, 
        stratify=np.argmax(y_right, axis=1), random_state=RANDOM_SEED
    )
    X_train, X_val, yr_train, yr_val, yl_train, yl_val = train_test_split(
        X_temp, yr_temp, yl_temp, test_size=VAL_RATIO, 
        stratify=np.argmax(yr_temp, axis=1), random_state=RANDOM_SEED
    )

    print(f"📊 Dataset Split → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Build Multi-Output Model
    model = build_model(
        sequence_length=SEQUENCE_LENGTH,
        feature_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES
    )
    model.summary()

    # 4. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

        # 5. Training (Dual Output)
    print("🏋️ Starting multi-output training...")
    history = model.fit(
    X_train, [yr_train, yl_train],
    validation_data=(X_val, [yr_val, yl_val]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
    )

    # 6. Post-Training
    plot_training_history(history, HISTORY_PLOT_PATH)
    np.save(LABEL_MAP_PATH, ACTIONS)
    print("✅ Label map saved to", LABEL_MAP_PATH)

    # Save Config & Best Metrics
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    best_val_loss = min(history.history.get('val_loss', [float('inf')]))
    
    config = {
        "model_path": MODEL_SAVE_PATH,
        "actions": ACTIONS,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_ratio": TEST_RATIO,
        "val_ratio": VAL_RATIO,
        "random_seed": RANDOM_SEED,
        "best_val_accuracy": float(best_val_acc),
        "best_val_loss": float(best_val_loss)
    }
    with open(CONFIG_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Config & metrics saved to {CONFIG_SAVE_PATH}")
    print(f"🏆 Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"📉 Best Val Loss: {best_val_loss:.4f}")
    print("🎉 Multi-output training complete! Jalankan 'python evaluate_multi.py' untuk testing final.")
    return history

# =========================
# RUN
# =========================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Training gagal: {e}")
        import traceback
        traceback.print_exc()