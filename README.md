# INFORME: Sistema de Reconocimiento Facial para Control de Acceso: PCA (Eigenfaces) vs LDA (Fisherfaces)

**Álgebra Lineal Aplicada 2026-I**  
Daniela Eléspuru

---

## Tabla de Contenidos

1. [Resumen](#1-resumen)
2. [Estructura del Repositorio](#2-estructura-del-repositorio)
3. [Fundamentos Matemáticos](#3-fundamentos-matemáticos)
   - 3.1 [Representación matricial de imágenes](#31-representación-matricial-de-imágenes)
   - 3.2 [PCA vía SVD — Eigenfaces](#32-pca-vía-svd--eigenfaces)
   - 3.3 [LDA — Fisherfaces](#33-lda--fisherfaces)
   - 3.4 [Clasificación por distancia euclidiana](#34-clasificación-por-distancia-euclidiana)
4. [Metodología](#4-metodología)
5. [Implementación](#5-implementación)
6. [Resultados y Benchmarking](#6-resultados-y-benchmarking)
7. [Conclusiones](#7-conclusiones)
8. [Instalación y Uso](#8-instalación-y-uso)
9. [Referencias](#9-referencias)

---

## 1. Resumen

Este proyecto implementa desde cero (exclusivamente NumPy) un sistema de autenticación facial que compara dos métodos de reducción dimensional:

| Método | Base matemática | Dimensiones | Accuracy global |
|--------|----------------|-------------|-----------------|
| **PCA (Eigenfaces)** | SVD / eigenvalores de covarianza | D=4096 → k=30 | **75.0 %** |
| **LDA (Fisherfaces)** | Discriminantes de Fisher | D=4096 → k=4 | **85.0 %** |

**Conclusión principal:** LDA supera a PCA en 10 puntos porcentuales, especialmente en la detección de intrusos (100 % vs 50 %), mientras que PCA es más robusto en la identificación de personas autorizadas (100 % vs 70 %). La reducción dimensional es drástica en ambos casos: PCA retiene el 99.48 % de la varianza con solo 30 de 4096 dimensiones; LDA opera con apenas 4 dimensiones.

---

## 2. Estructura del Repositorio

```
facial_recognition/
├── generate_images.py           # Generador de imágenes sintéticas
├── main_pipeline.py             # Pipeline principal (pasos 1–7)
├── visualize_and_benchmark.py   # Visualizaciones y benchmarking
├── requirements.txt
├── src/
│   └── face_recognition_core.py # PCAModel, LDAModel, utilidades
├── data/
│   ├── authorized/              # 5 personas × 10 imágenes = 50 imgs entrenamiento
│   │   ├── persona_01/ (10 PNGs)
│   │   ├── persona_02/ (10 PNGs)
│   │   ├── persona_03/ (10 PNGs)
│   │   ├── persona_04/ (10 PNGs)
│   │   └── persona_05/ (10 PNGs)
│   └── test/                    # 20 imágenes de prueba + labels
│       ├── test_persona_0X_vY.png  (10 imgs — 2 por persona autorizada)
│       ├── test_intruso_XX.png     (10 imgs — intrusos)
│       └── test_labels.csv
└── results/
    ├── evaluation_results.csv
    ├── summary_metrics.json
    ├── benchmarking.json
    └── figures/
        ├── fig1_eigenfaces.png
        ├── fig2_variance_pca.png
        ├── fig3_svd_reconstruction.png
        ├── fig4_pca_projection.png
        ├── fig5_lda_projection.png
        ├── fig6_confusion_matrices.png
        ├── fig7_benchmarking.png
        ├── fig8_accuracy_comparison.png
        └── fig9_authorized_gallery.png
```

---

## 3. Fundamentos Matemáticos

### 3.1 Representación matricial de imágenes

Cada imagen facial de dimensiones $64 \times 64$ píxeles (escala de grises) se convierte en un vector columna:

$$\mathbf{x} \in \mathbb{R}^{D}, \quad D = 64 \times 64 = 4096$$

La matriz de datos del conjunto de entrenamiento (N=50 imágenes) es:

$$X \in \mathbb{R}^{N \times D} = \mathbb{R}^{50 \times 4096}$$

### 3.2 PCA vía SVD — Eigenfaces

**Objetivo:** Encontrar la base ortonormal $\{w_1, \ldots, w_k\}$ que maximiza la varianza proyectada.

**Paso 1 — Centrado:**

$$\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{x}_i, \qquad X_c = X - \mathbf{1}\bar{\mathbf{x}}^\top$$

**Paso 2 — SVD económica:**

$$X_c = U \Sigma V^\top$$

donde $U \in \mathbb{R}^{N \times N}$, $\Sigma \in \mathbb{R}^{N \times N}$ (diagonal), $V \in \mathbb{R}^{D \times N}$.

Los vectores columna de $V$ son los eigenvectores de la matriz de covarianza $C = \frac{1}{N}X_c^\top X_c$, conocidos como **eigenfaces**.

**Paso 3 — Selección de componentes:**

$$W_{PCA} = V_{[:,\, 1:k]} \in \mathbb{R}^{D \times k}$$

La varianza explicada por el componente $i$-ésimo es:

$$\rho_i = \frac{\sigma_i^2}{\sum_{j=1}^{N} \sigma_j^2}$$

Con $k=30$ se retiene el **99.48 %** de la varianza total.

**Paso 4 — Proyección:**

$$Z = X_c \cdot W_{PCA} \in \mathbb{R}^{N \times k}$$

**Paso 5 — Clasificación:**

$$\hat{y} = \arg\min_{j} \|\mathbf{z}_{test} - \mathbf{z}_j\|_2$$

Si $\|\mathbf{z}_{test} - \mathbf{z}_{\hat{y}}\|_2 > \tau_{PCA}$, se deniega el acceso (intruso).

### 3.3 LDA — Fisherfaces

**Objetivo:** Maximizar la separabilidad entre clases, no la varianza total. El criterio de Fisher es:

$$J(W) = \frac{W^\top S_B W}{W^\top S_W W} \longrightarrow \max$$

donde $S_B$ es la dispersión **entre** clases (between-class scatter) y $S_W$ la dispersión **dentro** de cada clase (within-class scatter):

$$S_B = \sum_{c=1}^{C} n_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^\top$$

$$S_W = \sum_{c=1}^{C} \sum_{\mathbf{x} \in \mathcal{X}_c} (\mathbf{x} - \boldsymbol{\mu}_c)(\mathbf{x} - \boldsymbol{\mu}_c)^\top$$

**Problema de singularidad:** En espacio de píxeles, $S_W \in \mathbb{R}^{D \times D}$ es singular porque $D \gg N$. La solución **Fisherfaces** (Belhumeur et al., 1997) aplica primero PCA para reducir a $\mathbb{R}^{N-C}$ donde $S_W$ es invertible:

1. Reducir con PCA a $M = N - C = 45$ dimensiones
2. Calcular $S_B^{PCA}$ y $S_W^{PCA}$ en el espacio PCA
3. Resolver el problema de autovalores generalizado:

$$S_B^{PCA} \mathbf{w} = \lambda S_W^{PCA} \mathbf{w} \iff (S_W^{PCA})^{-1} S_B^{PCA} \mathbf{w} = \lambda \mathbf{w}$$

4. Los primeros $C-1 = 4$ eigenvectores (discriminantes de Fisher) forman $W_{Fisher}$
5. Proyección combinada: $W_{LDA} = W_{PCA} \cdot W_{Fisher} \in \mathbb{R}^{D \times 4}$

### 3.4 Clasificación por distancia euclidiana

Ambos métodos comparten el mismo clasificador de vecino más cercano:

$$d(\mathbf{x}_{test}, \mathbf{x}_j) = \sqrt{\sum_{i=1}^{k}(z_{test,i} - z_{j,i})^2}$$

---

## 4. Metodología

### Conjunto de datos

Se generaron imágenes sintéticas parametrizadas con características únicas por persona (tono de piel, separación ocular, tamaño de pupila, anchura nasal, forma de boca). Se añadió ruido gaussiano $\mathcal{N}(0, 6)$ para simular variabilidad de iluminación y posición.

| Conjunto | N | Descripción |
|----------|---|-------------|
| Entrenamiento | 50 | 5 personas × 10 imágenes cada una |
| Prueba | 20 | 10 de personas autorizadas (2 por persona) + 10 intrusos |

Todas las imágenes: 64×64 píxeles, escala de grises, normalizadas en [0,1].

### Flujo del sistema

```
[Imágenes autorizadas]
        │
        ▼
  Centrado (media facial)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
   SVD → eigenfaces W_PCA             PCA → W_PCA → S_B, S_W
   proyección Z_PCA                   SVD(S_W⁻¹S_B) → W_Fisher
   calibración τ_PCA                  W_LDA = W_PCA·W_Fisher
                                      proyección Z_LDA
                                      calibración τ_LDA
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
             [Imagen de prueba]
                       │
          ┌────────────┴────────────┐
          ▼                        ▼
   z_PCA = (x-μ)·W_PCA    z_LDA = (x-μ)·W_LDA
   d = min ‖z_PCA - Z_i‖  d = min ‖z_LDA - Z_i‖
   d > τ_PCA → DENEGADO    d > τ_LDA → DENEGADO
          │                        │
          └────────────┬───────────┘
                       ▼
            Comparación de accuracy
```

---

## 5. Implementación

Toda la implementación usa únicamente **NumPy** (sin scikit-learn para los algoritmos centrales).

### `PCAModel` — Eigenfaces

```python
# SVD económica — núcleo del método
X_c = X - self.mean_face           # centrado
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
self.eigenfaces = Vt[:k].T         # (D, k) — primeros k eigenfaces
self.projections = X_c @ self.eigenfaces  # (N, k) — galería proyectada
```

### `LDAModel` — Fisherfaces

```python
# Paso 1: PCA previo (evita singularidad de S_W)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
W_pca = Vt[:M].T                   # reducción a N-C dimensiones
X_pca = X_c @ W_pca

# Paso 2: Scatter matrices en espacio PCA
for c in classes:
    diff = X_pca[y==c] - mu_c
    S_W += diff.T @ diff
    S_B += n_c * np.outer(mu_c - mu_total, mu_c - mu_total)

# Paso 3: Problema de autovalores generalizado
S_W_inv = np.linalg.inv(S_W + 1e-8 * I)   # regularización Tikhonov
eigenvalues, eigenvectors = np.linalg.eig(S_W_inv @ S_B)
W_fisher = eigenvectors[:, top_k]

# Paso 4: Proyección combinada
self.W = W_pca @ W_fisher          # (D, k)
```

---

## 6. Resultados y Benchmarking

### Accuracy

| Métrica | PCA (k=30) | LDA (k=4) |
|---------|-----------|----------|
| Accuracy global | **75.0 %** | **85.0 %** |
| Accuracy — autorizados | 100.0 % | 70.0 % |
| Accuracy — intrusos | 50.0 % | 100.0 % |
| Umbral calibrado (τ) | 5.823 | 0.063 |

**PCA** identifica correctamente a todos los usuarios autorizados pero acepta falsamente algunos intrusos (umbral más permisivo en espacio de alta dimensión).  
**LDA** rechaza todos los intrusos pero requiere mayor precisión en el espacio discriminante, siendo más sensible a variaciones de iluminación en las imágenes de personas autorizadas.

### Benchmarking (n=50 repeticiones)

| Operación | PCA | LDA |
|-----------|-----|-----|
| Entrenamiento (μ ± σ) | 12.7 ± 0.4 ms | 14.6 ± 0.3 ms |
| Inferencia por imagen (μ ± σ) | 0.136 ± 0.028 ms | 0.039 ± 0.016 ms |

LDA es **3.5× más rápido** en inferencia porque opera en 4 dimensiones vs 30 de PCA.

### Reducción dimensional

| Método | D original | D reducida | Reducción |
|--------|-----------|-----------|-----------|
| PCA | 4096 | 30 | 99.3 % |
| LDA | 4096 | 4 | 99.9 % |

### Figuras generadas

| Figura | Descripción |
|--------|-------------|
| `fig1_eigenfaces.png` | Los 15 primeros eigenfaces (componentes principales) |
| `fig2_variance_pca.png` | Varianza acumulada explicada por PCA |
| `fig3_svd_reconstruction.png` | Reconstrucción con distintos rangos k |
| `fig4_pca_projection.png` | Scatter 2D en espacio PCA (PC1 vs PC2) |
| `fig5_lda_projection.png` | Scatter 2D en espacio LDA (LD1 vs LD2) |
| `fig6_confusion_matrices.png` | Matrices de confusión PCA y LDA |
| `fig7_benchmarking.png` | Boxplots de tiempos de entrenamiento e inferencia |
| `fig8_accuracy_comparison.png` | Comparación de métricas por método |
| `fig9_authorized_gallery.png` | Galería de personas autorizadas + rostro medio |

---

## 7. Conclusiones

1. **LDA supera a PCA en accuracy global** (85 % vs 75 %) para este problema de reconocimiento facial con 5 clases. Esto confirma la teoría: LDA optimiza la discriminabilidad entre clases, mientras que PCA maximiza la varianza total sin considerar la estructura de clases.

2. **PCA es superior en reconocimiento de autorizados** (100 % vs 70 %): su umbral en el espacio de eigenfaces es más flexible para absorber variaciones intra-clase, resultado de operar en 30 dimensiones en lugar de 4.

3. **LDA es superior en detección de intrusos** (100 % vs 50 %): los discriminantes de Fisher proyectan clases conocidas a regiones compactas y bien separadas; imágenes no pertenecientes a ninguna clase quedan naturalmente fuera de estas regiones.

4. **La reducción dimensional es drástica**: de 4096 dimensiones a 30 (PCA) o 4 (LDA), con pérdida de información mínima. SVD garantiza la mejor aproximación de rango reducido en norma de Frobenius (Teorema de Eckart-Young).

5. **LDA es más eficiente en inferencia** (0.039 ms vs 0.136 ms), lo que lo hace más adecuado para aplicaciones en tiempo real.

6. **La singularidad de S_W** — el principal desafío teórico de LDA en espacios de alta dimensión — se resuelve mediante la proyección PCA previa (enfoque Fisherfaces de Belhumeur et al., 1997), que garantiza que $S_W$ sea invertible en el subespacio reducido.

**Recomendación práctica:** Para un sistema de acceso donde el costo de un falso positivo (intruso aceptado) supera al de un falso negativo (autorizado rechazado), **LDA es el método preferido**. Si la prioridad es minimizar rechazos de usuarios autorizados, PCA ofrece mayor tolerancia.

---

## 8. Instalación y Uso

### Requisitos

```bash
pip install -r requirements.txt
```

### Ejecución paso a paso

```bash
# 1. Generar imágenes (si no están en data/)
python generate_images.py

# 2. Ejecutar pipeline principal
python main_pipeline.py

# 3. Generar visualizaciones y benchmarking
python visualize_and_benchmark.py
```

### Ejecutar todo de una vez

```bash
python generate_images.py && python main_pipeline.py && python visualize_and_benchmark.py
```

Los resultados se guardan en `results/` y las figuras en `results/figures/`.

---

## 9. Referencias

1. M. Turk y A. Pentland, "Eigenfaces for Recognition," *Journal of Cognitive Neuroscience*, vol. 3, no. 1, pp. 71–86, 1991.

2. I. T. Jolliffe y J. Cadima, "Principal component analysis: a review and recent developments," *Philosophical Transactions of the Royal Society A*, vol. 374, no. 2065, 2016.

3. J. Shlens, "A Tutorial on Principal Component Analysis," *arXiv preprint* arXiv:1404.1100, 2014.

4. F. L. Gewers et al., "Principal Component Analysis: A Natural Approach to Data Exploration," *arXiv preprint* arXiv:1804.02502, 2018.

5. A. Tharwat, T. Gaber, A. Ibrahim y A. E. Hassanien, “Linear Discriminant Analysis: A Detailed Tutorial,” AI Communications, vol. 30, no. 2, pp. 169–190, 2017.

6. Y. Li, Linear Discriminant Analysis and its Application to Face Identification, Ph.D. dissertation, University of Surrey, 2000.

