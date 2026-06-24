
import sys
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from face_recognition_core import (
    PCAModel, LDAModel,
    load_authorized_dataset, load_test_dataset,
    calibrate_threshold
)

DATA_DIR   = os.path.join(BASE_DIR, "data")
AUTH_DIR   = os.path.join(DATA_DIR, "authorized")
TEST_DIR   = os.path.join(DATA_DIR, "test")
FIG_DIR    = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

STYLE = {
    "bg": "#0f1117", "fg": "#e8eaf6", "accent1": "#7c83fd",
    "accent2": "#fc5c7d", "accent3": "#43e97b", "accent4": "#f7971e",
    "grid": "#2a2d3e"
}
plt.rcParams.update({
    "figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["bg"],
    "text.color": STYLE["fg"], "axes.labelcolor": STYLE["fg"],
    "xtick.color": STYLE["fg"], "ytick.color": STYLE["fg"],
    "axes.edgecolor": STYLE["grid"], "grid.color": STYLE["grid"],
    "font.family": "monospace", "font.size": 9,
})

COLORS = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"],
          STYLE["accent4"], "#a78bfa"]

# cargar datos y entrenar modelos 
print("[VIZ] Cargando datos...")
X_train, y_train, person_names = load_authorized_dataset(AUTH_DIR)
X_test,  y_test,  test_names   = load_test_dataset(TEST_DIR)

PCA_K = 30
LDA_K = 4

pca = PCAModel(n_components=PCA_K)
pca.fit(X_train, y_train)

lda = LDAModel(n_components=LDA_K)
lda.fit(X_train, y_train)

pca_threshold = calibrate_threshold(pca, X_train, y_train)
lda_threshold = calibrate_threshold(lda, X_train, y_train)

# prwdicciones
pca_preds, lda_preds = [], []
for x in X_test:
    pp, _ = pca.predict(x, threshold=pca_threshold)
    lp, _ = lda.predict(x, threshold=lda_threshold)
    pca_preds.append(pp)
    lda_preds.append(lp)

pca_preds = np.array(pca_preds)
lda_preds = np.array(lda_preds)

# grafico 1

def compute_confusion(preds, trues, n_classes):
    C = n_classes + 1
    cm = np.zeros((C, C), dtype=int)
    label_map = {i: i for i in range(n_classes)}
    label_map[-1] = n_classes
    for p, t in zip(preds, trues):
        cm[label_map[t], label_map[p]] += 1
    return cm

pca_cm = compute_confusion(pca_preds, y_test, len(person_names))
lda_cm = compute_confusion(lda_preds, y_test, len(person_names))
tick_labels = [f"P{i+1}" for i in range(len(person_names))] + ["INT"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, cm, title in [(axes[0], pca_cm, "PCA (Eigenfaces)"),
                       (axes[1], lda_cm, "LDA (Fisherfaces)")]:
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else STYLE["fg"]
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {title}", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("Comparación de Matrices de Confusión: PCA vs LDA",
             color=STYLE["fg"], fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_confusion_matrices.png"), dpi=150,
            bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
print("Figura guardada")

# grafico 2 
auth_mask     = (y_test >= 0)
intruder_mask = (y_test < 0)

metrics = {
    "PCA": {
        "Global":      np.mean(pca_preds == y_test) * 100,
        "Autorizados": np.mean(pca_preds[auth_mask] == y_test[auth_mask]) * 100,
        "Intrusos":    np.mean(pca_preds[intruder_mask] == y_test[intruder_mask]) * 100,
    },
    "LDA": {
        "Global":      np.mean(lda_preds == y_test) * 100,
        "Autorizados": np.mean(lda_preds[auth_mask] == y_test[auth_mask]) * 100,
        "Intrusos":    np.mean(lda_preds[intruder_mask] == y_test[intruder_mask]) * 100,
    }
}

categories = ["Global", "Autorizados", "Intrusos"]
x = np.arange(len(categories))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5))
bars_pca = ax.bar(x - width/2, [metrics["PCA"][c] for c in categories],
                   width, label="PCA (Eigenfaces)", color=STYLE["accent1"], alpha=0.85)
bars_lda = ax.bar(x + width/2, [metrics["LDA"][c] for c in categories],
                   width, label="LDA (Fisherfaces)", color=STYLE["accent2"], alpha=0.85)

for bars in [bars_pca, bars_lda]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f"{h:.0f}%", ha="center", fontsize=9, color=STYLE["fg"])

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 115)
ax.set_title("Comparación de Accuracy: PCA vs LDA\n(Conjunto de prueba: 20 imágenes)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis="y")
ax.axhline(100, color=STYLE["grid"], ls="--", lw=0.7)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig8_accuracy_comparison.png"), dpi=150,
            bbox_inches="tight", facecolor=STYLE["bg"])
plt.close()
print("Figura guardada")

# benchmarking
n_reps = 50
fit_times_pca, fit_times_lda = [], []
infer_times_pca, infer_times_lda = [], []
x_sample = X_test[0]

for _ in range(n_reps):
    t0 = time.perf_counter()
    _p = PCAModel(n_components=PCA_K); _p.fit(X_train, y_train)
    fit_times_pca.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    _l = LDAModel(n_components=LDA_K); _l.fit(X_train, y_train)
    fit_times_lda.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter(); pca.predict(x_sample)
    infer_times_pca.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter(); lda.predict(x_sample)
    infer_times_lda.append((time.perf_counter() - t0) * 1000)

bench_summary = {
    "pca_fit_mean_ms":   float(np.mean(fit_times_pca)),
    "pca_fit_std_ms":    float(np.std(fit_times_pca)),
    "lda_fit_mean_ms":   float(np.mean(fit_times_lda)),
    "lda_fit_std_ms":    float(np.std(fit_times_lda)),
    "pca_infer_mean_ms": float(np.mean(infer_times_pca)),
    "pca_infer_std_ms":  float(np.std(infer_times_pca)),
    "lda_infer_mean_ms": float(np.mean(infer_times_lda)),
    "lda_infer_std_ms":  float(np.std(infer_times_lda)),
    "n_repetitions": n_reps,
}
with open(os.path.join(BASE_DIR, "results", "benchmarking.json"), "w") as f:
    json.dump(bench_summary, f, indent=2)

print(f"\n  PCA entrenamiento : {np.mean(fit_times_pca):.1f} ± {np.std(fit_times_pca):.1f} ms")
print(f"  LDA entrenamiento : {np.mean(fit_times_lda):.1f} ± {np.std(fit_times_lda):.1f} ms")
print(f"  PCA inferencia    : {np.mean(infer_times_pca):.3f} ± {np.std(infer_times_pca):.3f} ms")
print(f"  LDA inferencia    : {np.mean(infer_times_lda):.3f} ± {np.std(infer_times_lda):.3f} ms")
print(f"\n[OK] Figuras en: {FIG_DIR}")
print("Benchmarking en: results/benchmarking.json")
