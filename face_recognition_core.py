
import numpy as np
import os
from PIL import Image

IMG_SIZE = 64 

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float64).flatten()
    return arr / 255.0


def load_authorized_dataset(data_dir: str):
    labels = sorted(os.listdir(data_dir))
    labels = [l for l in labels if os.path.isdir(os.path.join(data_dir, l))]
    X, y = [], []
    for idx, person in enumerate(labels):
        person_dir = os.path.join(data_dir, person)
        for fname in sorted(os.listdir(person_dir)):
            if fname.endswith(".png"):
                vec = load_image(os.path.join(person_dir, fname))
                X.append(vec)
                y.append(idx)
    return np.array(X), np.array(y), labels


def load_test_dataset(data_dir: str):
    import csv
    meta = []
    with open(os.path.join(data_dir, "test_labels.csv"), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta.append((row["filename"], int(row["true_label"]), row["person_name"]))

    X, y, names = [], [], []
    for fname, label, name in meta:
        vec = load_image(os.path.join(data_dir, fname))
        X.append(vec)
        y.append(label)
        names.append(name)
    return np.array(X), np.array(y), names


#pca 
class PCAModel:
    def __init__(self, n_components: int = 40):
        self.k = n_components
        self.mean_face: np.ndarray | None = None
        self.eigenfaces: np.ndarray | None = None      # (D, k)
        self.singular_values: np.ndarray | None = None
        self.projections: np.ndarray | None = None     # (N, k) — galería de autorizados
        self.labels: np.ndarray | None = None
        self.variance_explained: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, D = X.shape
        self.mean_face = X.mean(axis=0)
        X_c = X - self.mean_face  # centrado

        # SVD
        U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

        self.singular_values = s
        self.eigenfaces = Vt[:self.k].T  # (D, k)  — primeros k eigenfaces

        # var
        total_var = np.sum(s ** 2)
        self.variance_explained = (s ** 2) / total_var

        self.projections = X_c @ self.eigenfaces  # (N, k)
        self.labels = y

    def project(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_face) @ self.eigenfaces

    def predict(self, x: np.ndarray, threshold: float = None) -> tuple:
        z = self.project(x)
        dists = np.linalg.norm(self.projections - z, axis=1)
        idx_min = int(np.argmin(dists))
        d_min = float(dists[idx_min])
        pred_label = int(self.labels[idx_min])
        if threshold is not None and d_min > threshold:
            pred_label = -1
        return pred_label, d_min

    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.variance_explained)


# lda
class LDAModel:
    def __init__(self, n_components: int = 4):
        self.k = n_components
        self.mean_face: np.ndarray | None = None
        self.W: np.ndarray | None = None             # proyec total
        self.projections: np.ndarray | None = None   
        self.labels: np.ndarray | None = None
        self._pca_W: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, D = X.shape
        classes = np.unique(y)
        C = len(classes)
        self.mean_face = X.mean(axis=0)

        X_c = X - self.mean_face
        _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        M = min(N - C, D - 1)
        W_pca = Vt[:M].T          # (D, M)
        X_pca = X_c @ W_pca       # (N, M)
        self._pca_W = W_pca
        self._pca_mean = self.mean_face

        mu_total = X_pca.mean(axis=0)  # (M,)

        S_W = np.zeros((M, M))
        S_B = np.zeros((M, M))

        for c in classes:
            X_c_class = X_pca[y == c]
            n_c = X_c_class.shape[0]
            mu_c = X_c_class.mean(axis=0)

            diff = X_c_class - mu_c
            S_W += diff.T @ diff

            delta = (mu_c - mu_total).reshape(-1, 1)
            S_B += n_c * (delta @ delta.T)

        S_W_inv = np.linalg.inv(S_W + np.eye(M) * 1e-8)  # regularización leve
        A = S_W_inv @ S_B

        eigenvalues, eigenvectors = np.linalg.eig(A)

        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        idx_sorted = np.argsort(eigenvalues)[::-1]
        k_actual = min(self.k, C - 1)
        W_fisher = eigenvectors[:, idx_sorted[:k_actual]]  # (M, k)

        self.W = W_pca @ W_fisher      # (D, k)
        self.projections = X_c @ self.W  # (N, k)  — centrado respecto a mean_face original
        X_orig_c = X - self.mean_face
        self.projections = X_orig_c @ self.W
        self.labels = y

    def project(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_face) @ self.W

    def predict(self, x: np.ndarray, threshold: float = None) -> tuple:
        z = self.project(x)
        dists = np.linalg.norm(self.projections - z, axis=1)
        idx_min = int(np.argmin(dists))
        d_min = float(dists[idx_min])
        pred_label = int(self.labels[idx_min])
        if threshold is not None and d_min > threshold:
            pred_label = -1
        return pred_label, d_min

# calibracion

def calibrate_threshold(model, X_train: np.ndarray, y_train: np.ndarray,
                         percentile: float = 95.0) -> float:
    if isinstance(model, PCAModel):
        projections = model.projections  # (N, k)
    else:
        projections = model.projections  # (N, k)

    classes = np.unique(y_train)
    dists = []

    for i in range(len(X_train)):
        # Proyección de la imagen i
        z_i = projections[i]
        y_i = y_train[i]

        # Distancias a imágenes de la misma clase (excluyendo i misma)
        same_class = (y_train == y_i)
        same_class[i] = False  # excluir i misma

        if same_class.sum() == 0:
            continue

        z_others = projections[same_class]
        d_min = float(np.min(np.linalg.norm(z_others - z_i, axis=1)))
        dists.append(d_min)

    if len(dists) == 0:
        return 1e-3  # fallback

    base = float(np.percentile(dists, percentile))
   