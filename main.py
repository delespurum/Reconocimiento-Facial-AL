import sys
import os
import time
import numpy as np

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
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# calculos
PCA_COMPONENTS = 30    # eigenfaces a retener
LDA_COMPONENTS = 4     # C-1 = 4 discriminantes de Fisher

# lectura de autorizados
X_train, y_train, person_names = load_authorized_dataset(AUTH_DIR)
N, D = X_train.shape
C = len(person_names)

print(f"  → {N} imágenes cargadas | {C} personas | D = {D} dimensiones")
print(f"  → Personas: {', '.join(person_names)}")

#pca y svd
t0 = time.perf_counter()
pca = PCAModel(n_components=PCA_COMPONENTS)
pca.fit(X_train, y_train)
t_pca_fit = time.perf_counter() - t0

# var acumulada
cum_var = pca.cumulative_variance()
var_k = cum_var[PCA_COMPONENTS - 1] * 100
print(f"  → {PCA_COMPONENTS} componentes retienen {var_k:.2f}% de la varianza")
print(f"  → Tiempo de entrenamiento PCA: {t_pca_fit*1000:.2f} ms")

# lda
t0 = time.perf_counter()
lda = LDAModel(n_components=LDA_COMPONENTS)
lda.fit(X_train, y_train)
t_lda_fit = time.perf_counter() - t0
print(f"  → {LDA_COMPONENTS} discriminantes de Fisher calculados")
print(f"  → Tiempo de entrenamiento LDA: {t_lda_fit*1000:.2f} ms")

# calibracion
pca_threshold = calibrate_threshold(pca, X_train, y_train, percentile=95)
lda_threshold = calibrate_threshold(lda, X_train, y_train, percentile=95)

print(f"  → Umbral PCA (percentil 95): {pca_threshold:.6f}")
print(f"  → Umbral LDA (percentil 95): {lda_threshold:.6f}")

print("\n  [Galería PCA almacenada]")
print(f"  Shape proyecciones: {pca.projections.shape}")
print("\n  [Galería LDA almacenada]")
print(f"  Shape proyecciones: {lda.projections.shape}")

# loop testing
X_test, y_test, test_names = load_test_dataset(TEST_DIR)

pca_preds, lda_preds = [], []
pca_dists, lda_dists = [], []
pca_times, lda_times = [], []

AUTHORIZED_IDX = list(range(C))   

header = f"{'#':>3} | {'Archivo':<30} | {'Real':<12} | {'PCA':<12} | {'LDA':<12} | {'AccPCA':>6} | {'AccLDA':>6}"
print(header)
print("-" * len(header))

for i, (x, y_true, name) in enumerate(zip(X_test, y_test, test_names)):
    # PCA
    t0 = time.perf_counter()
    pca_label, pca_dist = pca.predict(x, threshold=pca_threshold)
    pca_times.append(time.perf_counter() - t0)
    pca_preds.append(pca_label)
    pca_dists.append(pca_dist)

    # LDA
    t0 = time.perf_counter()
    lda_label, lda_dist = lda.predict(x, threshold=lda_threshold)
    lda_times.append(time.perf_counter() - t0)
    lda_preds.append(lda_label)
    lda_dists.append(lda_dist)

    #labels 
    y_true_name = person_names[y_true] if y_true >= 0 else "intruso"
    pca_pred_name = person_names[pca_label] if pca_label >= 0 else "DENEGADO"
    lda_pred_name = person_names[lda_label] if lda_label >= 0 else "DENEGADO"

    pca_ok = "✓" if pca_label == y_true else "✗"
    lda_ok = "✓" if lda_label == y_true else "✗"

    print(f"{i+1:>3} | {name:<30} | {y_true_name:<12} | {pca_pred_name:<12} | {lda_pred_name:<12} | {pca_ok:>6} | {lda_ok:>6}")

print("-" * len(header))

# final
pca_preds = np.array(pca_preds)
lda_preds = np.array(lda_preds)
y_test    = np.array(y_test)

pca_acc = np.mean(pca_preds == y_test)
lda_acc = np.mean(lda_preds == y_test)

# autorizados vs intrusos
auth_mask    = (y_test >= 0)
intruder_mask = (y_test == -1)

pca_acc_auth    = np.mean(pca_preds[auth_mask] == y_test[auth_mask])
pca_acc_intruder = np.mean(pca_preds[intruder_mask] == y_test[intruder_mask])
lda_acc_auth    = np.mean(lda_preds[auth_mask] == y_test[auth_mask])
lda_acc_intruder = np.mean(lda_preds[intruder_mask] == y_test[intruder_mask])

print("\n[RESULTADOS FINALES]")
print(f"\n  MÉTODO PCA (Eigenfaces, k={PCA_COMPONENTS}):")
print(f"    Accuracy global          : {pca_acc*100:.1f}%")
print(f"    Accuracy en autorizados  : {pca_acc_auth*100:.1f}%")
print(f"    Accuracy en intrusos     : {pca_acc_intruder*100:.1f}%")
print(f"    Tiempo promedio inferencia: {np.mean(pca_times)*1000:.3f} ms")

print(f"\n  MÉTODO LDA (Fisherfaces, k={LDA_COMPONENTS}):")
print(f"    Accuracy global          : {lda_acc*100:.1f}%")
print(f"    Accuracy en autorizados  : {lda_acc_auth*100:.1f}%")
print(f"    Accuracy en intrusos     : {lda_acc_intruder*100:.1f}%")
print(f"    Tiempo promedio inferencia: {np.mean(lda_times)*1000:.3f} ms")

print("\n[CONCLUSIÓN]")
if lda_acc > pca_acc:
    winner = "LDA (Fisherfaces)"
    diff = (lda_acc - pca_acc) * 100
    reason = "LDA maximiza la separación entre clases (ratio S_B / S_W), siendo más discriminativo que PCA que solo maximiza varianza total."
elif pca_acc > lda_acc:
    winner = "PCA (Eigenfaces)"
    diff = (pca_acc - lda_acc) * 100
    reason = "PCA captura más variabilidad global con mayor cantidad de componentes, siendo más robusto ante imágenes ruidosas en este dataset reducido."
else:
    winner = "Empate"
    diff = 0.0
    reason = "Ambos métodos tienen rendimiento equivalente en este conjunto de prueba."

print(f"  → Método más preciso: {winner}")
if diff > 0:
    print(f"  → Ventaja: {diff:.1f} puntos porcentuales")
print(f"  → Justificación: {reason}")

print(f"\n  Dimensionalidad: {D} → PCA: {PCA_COMPONENTS} | LDA: {LDA_COMPONENTS}")
print(f"  Reducción PCA: {(1 - PCA_COMPONENTS/D)*100:.1f}% | LDA: {(1 - LDA_COMPONENTS/D)*100:.1f}%")

# resultados 
import csv
with open(os.path.join(RESULT_DIR, "evaluation_results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["idx", "filename", "true_label", "pca_pred", "lda_pred",
                     "pca_dist", "lda_dist", "pca_correct", "lda_correct"])
    for i, (name, y_true, pp, lp, pd, ld) in enumerate(
            zip(test_names, y_test, pca_preds, lda_preds, pca_dists, lda_dists)):
        writer.writerow([i+1, name, int(y_true), int(pp), int(lp),
                         f"{pd:.6f}", f"{ld:.6f}",
                         int(pp == y_true), int(lp == y_true)])

print(f"\n Resultados guardados en: {os.path.join(RESULT_DIR, 'evaluation_results.csv')}")

# resumen
summary = {
    "pca_components": PCA_COMPONENTS,
    "lda_components": LDA_COMPONENTS,
    "pca_accuracy": pca_acc,
    "lda_accuracy": lda_acc,
    "pca_acc_auth": pca_acc_auth,
    "pca_acc_intruder": pca_acc_intruder,
    "lda_acc_auth": lda_acc_auth,
    "lda_acc_intruder": lda_acc_intruder,
    "pca_threshold": pca_threshold,
    "lda_threshold": lda_threshold,
    "pca_fit_time_ms": t_pca_fit * 1000,
    "lda_fit_time_ms": t_lda_fit * 1000,
    "pca_infer_time_ms": float(np.mean(pca_times)) * 1000,
    "lda_infer_time_ms": float(np.mean(lda_times)) * 1000,
    "variance_explained_pca": float(var_k),
    "winner": winner,
}

import json
with open(os.path.join(RESULT_DIR, "summary_metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"Métricas guardadas en: {os.path.join(RESULT_DIR, 'summary_metrics.json')}")
print("\n" + "=" * 60)
