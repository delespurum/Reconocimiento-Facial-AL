import numpy as np
from PIL import Image, ImageDraw
import os

SEED = 42
IMG_SIZE = 64  # 64x64 pix (4096 dimensiones)

#parametros 
PERSON_PARAMS = {
    "persona_01": dict(skin=180, eye_sep=18, eye_sz=5, nose_w=7, mouth_w=16, face_h=38, brow_h=3),
    "persona_02": dict(skin=140, eye_sep=16, eye_sz=4, nose_w=6, mouth_w=14, face_h=36, brow_h=2),
    "persona_03": dict(skin=200, eye_sep=20, eye_sz=6, nose_w=8, mouth_w=18, face_h=40, brow_h=4),
    "persona_04": dict(skin=160, eye_sep=15, eye_sz=5, nose_w=5, mouth_w=12, face_h=34, brow_h=3),
    "persona_05": dict(skin=220, eye_sep=22, eye_sz=7, nose_w=9, mouth_w=20, face_h=42, brow_h=2),
}

INTRUDER_PARAMS = [
    dict(skin=130, eye_sep=12, eye_sz=3, nose_w=5, mouth_w=10, face_h=30, brow_h=1),
    dict(skin=190, eye_sep=25, eye_sz=8, nose_w=10, mouth_w=22, face_h=44, brow_h=5),
    dict(skin=155, eye_sep=14, eye_sz=4, nose_w=6, mouth_w=13, face_h=32, brow_h=2),
    dict(skin=210, eye_sep=23, eye_sz=6, nose_w=9, mouth_w=19, face_h=41, brow_h=4),
    dict(skin=135, eye_sep=13, eye_sz=3, nose_w=4, mouth_w=11, face_h=31, brow_h=1),
    dict(skin=175, eye_sep=19, eye_sz=5, nose_w=7, mouth_w=15, face_h=37, brow_h=3),
    dict(skin=195, eye_sep=21, eye_sz=7, nose_w=8, mouth_w=17, face_h=39, brow_h=4),
    dict(skin=145, eye_sep=17, eye_sz=4, nose_w=6, mouth_w=14, face_h=35, brow_h=2),
    dict(skin=165, eye_sep=16, eye_sz=5, nose_w=7, mouth_w=16, face_h=38, brow_h=3),
    dict(skin=215, eye_sep=24, eye_sz=7, nose_w=10, mouth_w=21, face_h=43, brow_h=5),
]


def draw_face(params: dict, noise_seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(noise_seed)
    size = IMG_SIZE
    cx, cy = size // 2, size // 2

    img = Image.new("L", (size, size), color=30)
    draw = ImageDraw.Draw(img)

    skin = params["skin"]
    face_h = params["face_h"]
    face_w = int(face_h * 0.82)

    # caras
    draw.ellipse(
        [cx - face_w, cy - face_h, cx + face_w, cy + face_h],
        fill=skin,
    )

    # ojos
    sep = params["eye_sep"]
    esz = params["eye_sz"]
    eye_y = cy - face_h // 4
    for ex in [cx - sep, cx + sep]:
        draw.ellipse([ex - esz, eye_y - esz, ex + esz, eye_y + esz], fill=30)
        draw.ellipse(
            [ex - esz + 1, eye_y - esz + 1, ex + esz - 1, eye_y + esz - 1], fill=200
        )
        draw.ellipse([ex - 1, eye_y - 1, ex + 1, eye_y + 1], fill=10)

    # cejas
    bh = params["brow_h"]
    brow_y = eye_y - esz - 4
    for bx in [cx - sep, cx + sep]:
        draw.rectangle(
            [bx - sep // 2, brow_y - bh, bx + sep // 2, brow_y], fill=max(0, skin - 60)
        )

    # nariz
    nw = params["nose_w"]
    nose_y = cy + face_h // 10
    draw.ellipse([cx - nw // 2, nose_y - 3, cx + nw // 2, nose_y + 3], fill=max(0, skin - 30))

    # boca
    mw = params["mouth_w"]
    mouth_y = cy + face_h // 3
    draw.arc([cx - mw, mouth_y - 5, cx + mw, mouth_y + 5], start=0, end=180, fill=max(0, skin - 80), width=2)

    arr = np.array(img, dtype=np.float32)

    # noise para añadir variabilidad 
    noise = rng.randn(*arr.shape) * 6
    arr = np.clip(arr + noise, 0, 255)

    return arr.astype(np.uint8)


def save_image(arr: np.ndarray, path: str):
    Image.fromarray(arr, mode="L").save(path)


def generate_authorized(output_dir: str, n_per_person: int = 10):
    os.makedirs(output_dir, exist_ok=True)
    for person_id, params in PERSON_PARAMS.items():
        person_dir = os.path.join(output_dir, person_id)
        os.makedirs(person_dir, exist_ok=True)
        for i in range(n_per_person):
            seed = SEED + hash(person_id) % 1000 + i
            arr = draw_face(params, noise_seed=seed)
            save_image(arr, os.path.join(person_dir, f"{person_id}_img{i+1:02d}.png"))
    print(f"Imágenes autorizadas generadas en: {output_dir}")


def generate_test(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    labels = []
    filenames = []

    persons = list(PERSON_PARAMS.keys())

    # 2 por persona autorizado 
    for idx, (person_id, params) in enumerate(PERSON_PARAMS.items()):
        for j in range(2):
            seed = 500 + idx * 10 + j
            arr = draw_face(params, noise_seed=seed)
            fname = f"test_{person_id}_v{j+1}.png"
            save_image(arr, os.path.join(output_dir, fname))
            labels.append(idx)  # clase = índice de persona
            filenames.append(fname)

    # 10 intrusos
    for k, iparams in enumerate(INTRUDER_PARAMS):
        seed = 900 + k * 7
        arr = draw_face(iparams, noise_seed=seed)
        fname = f"test_intruso_{k+1:02d}.png"
        save_image(arr, os.path.join(output_dir, fname))
        labels.append(-1)  # intruso
        filenames.append(fname)

   
    import csv
    with open(os.path.join(output_dir, "test_labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label", "person_name"])
        for fname, label in zip(filenames, labels):
            name = persons[label] if label >= 0 else "intruso"
            writer.writerow([fname, label, name])

    print(f"{len(filenames)} imágenes de prueba generadas en: {output_dir}")
    return filenames, labels


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    generate_authorized(os.path.join(base, "data", "authorized"))
    generate_test(os.path.join(base, "data", "test"))
    print("Generación de imágenes completada.")
