# evaluate_multi.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
from keras.utils import to_categorical

from model_multi import HandSplitter

# =========================
# CONFIG
# =========================
DATASET_PATH = "dataset"
MODEL_PATH = "models/gangsa_bilstm_multi.h5"
LABEL_MAP_PATH = "models/label_map.npy"
METRICS_SAVE_PATH = "models/evaluation_multi.json"

ACTIONS = ['pukul', 'redam', 'netral']
SEQ_LEN = 15
FEATURE_DIM = 128
NUM_CLASSES = len(ACTIONS)
TEST_RATIO = 0.2
RANDOM_SEED = 42

# =========================
# LOAD TEST DATA
# =========================
def load_test_data():
    print("📦 Loading dataset...")
    X, labels_r, labels_l = [], [], []

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Folder '{DATASET_PATH}' tidak ditemukan!")

    for action in ACTIONS:
        folder = os.path.join(DATASET_PATH, action)
        if not os.path.exists(folder): continue

        for file in os.listdir(folder):
            if file.endswith(".npy"):
                seq = np.load(os.path.join(folder, file))
                if seq.shape != (SEQ_LEN, FEATURE_DIM): continue
                X.append(seq)
                if action == 'pukul':
                    labels_r.append(0); labels_l.append(2)
                elif action == 'redam':
                    labels_r.append(2); labels_l.append(1)
                else:
                    labels_r.append(2); labels_l.append(2)

    if len(X) == 0:
        raise ValueError("Tidak ada data valid!")

    X = np.array(X)
    y_right = to_categorical(labels_r, num_classes=NUM_CLASSES)
    y_left  = to_categorical(labels_l, num_classes=NUM_CLASSES)

    _, X_test, _, yr_test, _, yl_test = train_test_split(
        X, y_right, y_left, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )
    print(f"✅ Test set loaded: {len(X_test)} sequences")
    return X_test, yr_test, yl_test

# =========================
# EVALUATE & VISUALIZE
# =========================
def run_evaluation():
    print(f"🧠 Loading model dari {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH, custom_objects={"HandSplitter": HandSplitter})
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).tolist()

    X_test, y_right_test, y_left_test = load_test_data()
    y_true_r = np.argmax(y_right_test, axis=1)
    y_true_l = np.argmax(y_left_test, axis=1)

    print("🔮 Generating predictions...")
    preds_right, preds_left = model.predict(X_test, verbose=0)
    y_pred_r = np.argmax(preds_right, axis=1)
    y_pred_l = np.argmax(preds_left, axis=1)

    # 1. Per-Hand Accuracy
    acc_r = accuracy_score(y_true_r, y_pred_r)
    acc_l = accuracy_score(y_true_l, y_pred_l)
    print(f"\n🎯 Right Hand Accuracy: {acc_r*100:.2f}%")
    print(f"🎯 Left  Hand Accuracy: {acc_l*100:.2f}%")

    # 2. Confusion Matrices & Reports (3x3 forced)
    all_labels = [0, 1, 2]
    cm_r = confusion_matrix(y_true_r, y_pred_r, labels=all_labels)
    cm_l = confusion_matrix(y_true_l, y_pred_l, labels=all_labels)

    report_r = classification_report(y_true_r, y_pred_r, labels=all_labels, 
                                     target_names=label_map, zero_division=0, output_dict=True)
    report_l = classification_report(y_true_l, y_pred_l, labels=all_labels, 
                                     target_names=label_map, zero_division=0, output_dict=True)

    # 🔑 EKSTRAK METRIK 'NETRAL' (Index 2)
    netral_r = report_r.get('2', report_r.get('netral', {}))
    netral_l = report_l.get('2', report_l.get('netral', {}))

    print("\n📊 METRIK KHUSUS KELAS 'NETRAL':")
    print(f"   🖐️ Kanan - Precision: {netral_r['precision']:.3f} | Recall: {netral_r['recall']:.3f} | F1: {netral_r['f1-score']:.3f}")
    print(f"   🖐️ Kiri   - Precision: {netral_r['precision']:.3f} | Recall: {netral_l['recall']:.3f} | F1: {netral_l['f1-score']:.3f}")

    # 🔑 OVERALL SYSTEM ACCURACY (Kedua tangan benar secara bersamaan)
    system_correct = np.sum((y_pred_r == y_true_r) & (y_pred_l == y_true_l))
    system_acc = system_correct / len(y_true_r)
    print(f"🌐 Overall System Accuracy (Both Hands Match): {system_acc*100:.2f}%")

    # 3. Plot Confusion Matrices
    os.makedirs("models", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm_r, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=label_map, yticklabels=label_map)
    ax1.set_title('Right Hand Confusion Matrix'); ax1.set_ylabel('True'); ax1.set_xlabel('Predicted')
    sns.heatmap(cm_l, annot=True, fmt='d', cmap='Reds', ax=ax2, xticklabels=label_map, yticklabels=label_map)
    ax2.set_title('Left Hand Confusion Matrix'); ax2.set_ylabel('True'); ax2.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix_multi.png', dpi=300)
    plt.close()
    print("✅ Confusion matrices saved.")

    # 4. Bar Chart Accuracy
    plt.figure(figsize=(7, 4))
    bars = plt.bar(['Right Hand', 'Left Hand', 'System (Both)'], 
                   [acc_r, acc_l, system_acc], color=['#4ECDC4', '#FF6B6B', '#6C5CE7'])
    plt.ylim(0, 1.05)
    plt.title('Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy')
    for bar, val in zip(bars, [acc_r, acc_l, system_acc]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val*100:.1f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/accuracy_comparison.png', dpi=300)
    plt.close()
    print("✅ Accuracy chart saved.")

    # 5. Save Detailed Metrics
    metrics = {
        "right_hand": {"accuracy": float(acc_r), "classification_report": report_r, "confusion_matrix": cm_r.tolist()},
        "left_hand": {"accuracy": float(acc_l), "classification_report": report_l, "confusion_matrix": cm_l.tolist()},
        "system_accuracy": float(system_acc),
        "netral_performance": {
            "right": {"precision": float(netral_r['precision']), "recall": float(netral_r['recall']), "f1": float(netral_r['f1-score'])},
            "left": {"precision": float(netral_l['precision']), "recall": float(netral_l['recall']), "f1": float(netral_l['f1-score'])}
        }
    }
    with open(METRICS_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Detailed metrics saved to {METRICS_SAVE_PATH}")

    return metrics

if __name__ == "__main__":
    try:
        run_evaluation()
        print("\n🎉 Evaluasi selesai! Cek folder 'models/' untuk hasil lengkap.")
    except Exception as e:
        print(f"\n❌ Evaluasi gagal: {e}")
        import traceback
        traceback.print_exc()