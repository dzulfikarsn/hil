from pathlib import Path
from datetime import datetime
import json
import yaml
import torch
import ultralytics
import pandas as pd


DATASET_NAME = "LeukemiaAttri_YOLO"
MODEL_NAME = "yolo11l.pt"

KAGGLE_INPUT = Path("/home/mf/.cache/kagglehub/datasets/albatani/leukemiaattri-yolo-train-val-test/versions/1")

OUTPUT_ROOT = Path(
    "/home/mf/Documents/jull/hil/hil2"
)

TRAIN_DIR = OUTPUT_ROOT / "training"
VAL_DIR = OUTPUT_ROOT / "validation"
TEST_DIR = OUTPUT_ROOT / "testing"
SUMMARY_DIR = OUTPUT_ROOT / "summary"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    SUMMARY_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

dataset_candidates = sorted([
    path
    for path in KAGGLE_INPUT.rglob(DATASET_NAME)
    if (
        path.is_dir()
        and (path / "data.yaml").is_file()
        and (path / "images").is_dir()
        and (path / "labels").is_dir()
    )
])

if not dataset_candidates:
    raise FileNotFoundError(
        f"Dataset '{DATASET_NAME}' tidak ditemukan "
        f"di {KAGGLE_INPUT}."
    )

if len(dataset_candidates) > 1:
    raise RuntimeError(
        "Ditemukan beberapa folder dataset yang cocok:\n"
        + "\n".join(map(str, dataset_candidates))
    )

DATASET_ROOT = dataset_candidates[0].resolve()
DATA_YAML_ORIGINAL = DATASET_ROOT / "data.yaml"

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"


print("=" * 65)
print("DATASET INFORMATION")
print("=" * 65)

print("Dataset root :", DATASET_ROOT)
print("Original YAML:", DATA_YAML_ORIGINAL)
print("Model        :", MODEL_NAME)
print("Device       :", DEVICE)

print("\nSYSTEM INFORMATION")
print("-" * 65)

print("CUDA available     :", torch.cuda.is_available())
print("Ultralytics version:", ultralytics.__version__)
print("PyTorch version    :", torch.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU tidak tersedia. Training akan menggunakan CPU")

with open(
    DATA_YAML_ORIGINAL,
    "r",
    encoding="utf-8"
) as file:
    data_config = yaml.safe_load(file)

if not isinstance(data_config, dict):
    raise ValueError(
        "Format data.yaml tidak valid"
    )

class_names = data_config.get("names")
num_classes = data_config.get("nc")

if class_names is None:
    raise ValueError(
        "Nama kelas 'names' tidak ditemukan di data.yaml"
    )

if isinstance(class_names, dict):
    try:
        class_names = {
            int(key): value
            for key, value in class_names.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Indeks kelas pada names harus berupa angka"
        ) from exc

    expected_ids = set(range(len(class_names)))

    if set(class_names.keys()) != expected_ids:
        raise ValueError(
            "Indeks kelas harus berurutan mulai dari 0"
        )

    class_names = dict(sorted(class_names.items()))

elif isinstance(class_names, list):
    pass

else:
    raise ValueError(
        "'names' harus berupa list atau dictionary"
    )

if not class_names:
    raise ValueError(
        "Daftar nama kelas kosong"
    )

if any(
    not isinstance(name, str) or not name.strip()
    for name in (
        class_names.values()
        if isinstance(class_names, dict)
        else class_names
    )
):
    raise ValueError(
        "Setiap nama kelas harus berupa teks yang tidak kosong"
    )

if num_classes is None:
    num_classes = len(class_names)
else:
    num_classes = int(num_classes)

if num_classes != len(class_names):
    raise ValueError(
        f"Jumlah kelas tidak cocok: nc={num_classes}, "
        f"jumlah names={len(class_names)}."
    )

data_config["nc"] = num_classes
data_config["names"] = class_names

print("\nCLASS CONFIGURATION")
print("-" * 65)

print("Number of classes:", num_classes)

for class_id, class_name in (
    class_names.items()
    if isinstance(class_names, dict)
    else enumerate(class_names)
):
    print(f"  {class_id}: {class_name}")


SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

dataset_counts = {}
image_stems_by_split = {}
label_stems_by_split = {}

print("\nVALIDATING DATASET STRUCTURE")
print("-" * 65)

for split_name in SPLITS:

    image_dir = IMAGES_DIR / split_name
    label_dir = LABELS_DIR / split_name

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: {image_dir}"
        )

    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"Folder label tidak ditemukan: {label_dir}"
        )

    image_files = sorted([
        path
        for path in image_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    ])

    label_files = sorted([
        path
        for path in label_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".txt"
    ])

    if not image_files:
        raise ValueError(
            f"Tidak ada gambar pada {image_dir}"
        )

    # Relative stems juga mendeteksi struktur subfolder.
    image_stems = {
        path.relative_to(image_dir).with_suffix("").as_posix()
        for path in image_files
    }

    label_stems = {
        path.relative_to(label_dir).with_suffix("").as_posix()
        for path in label_files
    }

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    image_stems_by_split[split_name] = image_stems
    label_stems_by_split[split_name] = label_stems

    dataset_counts[split_name] = {
        "images": len(image_files),
        "label_files": len(label_files),
        "images_without_label_file": len(missing_labels),
        "labels_without_image": len(orphan_labels)
    }

    print(f"\n{split_name.upper()}")
    print("Images directory :", image_dir)
    print("Labels directory :", label_dir)
    print("Total images     :", len(image_files))
    print("Total label files:", len(label_files))
    print("Images without label file:", len(missing_labels))
    print("Labels without image     :", len(orphan_labels))

    if missing_labels:
        print(
            "WARNING: Contoh gambar tanpa file label:",
            missing_labels[:5]
        )

    if orphan_labels:
        print(
            "WARNING: Contoh label tanpa gambar:",
            orphan_labels[:5]
        )


print("\nVALIDATING SPLIT OVERLAP")
print("-" * 65)

for i, split_a in enumerate(SPLITS):
    for split_b in SPLITS[i + 1:]:

        overlap = (
            image_stems_by_split[split_a]
            & image_stems_by_split[split_b]
        )

        print(
            f"{split_a} vs {split_b}: "
            f"{len(overlap)} matching relative filenames"
        )

        if overlap:
            print(
                "WARNING: Nama file relatif yang sama ditemukan"
                "Periksa apakah gambar yang sama masuk ke "
                "lebih dari satu split"
            )

data_config["path"] = str(DATASET_ROOT)

for split_name in SPLITS:
    data_config[split_name] = str(
        (IMAGES_DIR / split_name).resolve()
    )

DATA_YAML_FIXED = SUMMARY_DIR / "data_fixed.yaml"

with open(
    DATA_YAML_FIXED,
    "w",
    encoding="utf-8"
) as file:
    yaml.safe_dump(
        data_config,
        file,
        sort_keys=False,
        allow_unicode=True
    )

print("\nFIXED DATA.YAML")
print("-" * 65)

print("Saved to:", DATA_YAML_FIXED)
print(
    yaml.safe_dump(
        data_config,
        sort_keys=False,
        allow_unicode=True
    )
)

manifest_path = DATASET_ROOT / "dataset_manifest.csv"

dataset_info = {
    "dataset_name": DATASET_NAME,
    "dataset_root": str(DATASET_ROOT),
    "source_yaml": str(DATA_YAML_ORIGINAL),
    "resolved_yaml": str(DATA_YAML_FIXED),
    "model_name": MODEL_NAME,
    "number_of_classes": num_classes,
    "class_names": class_names,
    "train_path": data_config["train"],
    "validation_path": data_config["val"],
    "test_path": data_config["test"],
    "dataset_counts": dataset_counts,
    "manifest_path": (
        str(manifest_path)
        if manifest_path.is_file()
        else None
    ),
    "ultralytics_version": ultralytics.__version__,
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    ),
    "created_at": datetime.now().isoformat()
}

DATASET_INFO_FILE = SUMMARY_DIR / "dataset_info.json"

with open(
    DATASET_INFO_FILE,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        dataset_info,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 65)
print("DATASET CONFIGURATION SUMMARY")
print("=" * 65)

summary_df = pd.DataFrame([
    {
        "Split": split_name.upper(),
        "Images": dataset_counts[split_name]["images"],
        "Label Files": dataset_counts[split_name]["label_files"],
        "Images Without Label": dataset_counts[split_name][
            "images_without_label_file"
        ],
        "Labels Without Image": dataset_counts[split_name][
            "labels_without_image"
        ]
    }
    for split_name in SPLITS
])

# Print plain text instead of triggering Pandas HTML rendering.
print(summary_df.to_string(index=False))

print("\nDataset root :", DATASET_ROOT)
print("Fixed YAML   :", DATA_YAML_FIXED)
print("Dataset info :", DATASET_INFO_FILE)
print("Model        :", MODEL_NAME)
print("Device       :", DEVICE)

print("\nDataset configuration validated successfully!")
print("Ready for yolo11l training.")

from pathlib import Path
import torch
from ultralytics import YOLO


model = YOLO(MODEL_NAME)
train_results = model.train(
    data=str(DATA_YAML_FIXED),
    epochs=150,
    project=str(TRAIN_DIR),
    name="yolo11l",
    device=DEVICE,
    plots=True,
    save=True,
    exist_ok=True,
    verbose=True,
)

TRAIN_RUN_DIR = Path(train_results.save_dir)
WEIGHTS_DIR = TRAIN_RUN_DIR / "weights"

BEST_MODEL = WEIGHTS_DIR / "best.pt"
LAST_MODEL = WEIGHTS_DIR / "last.pt"

print("Training completed")
print("Model:", MODEL_NAME)
print("Training directory:", TRAIN_RUN_DIR)
print("Best model:", BEST_MODEL)
print("Last model:", LAST_MODEL)

assert BEST_MODEL.exists(), "The best model weights were not found"

best_model = YOLO(str(BEST_MODEL))

val_results = best_model.val(
    data=str(DATA_YAML_FIXED),
    split="val",
    device=DEVICE,
    plots=True,
    project=str(VAL_DIR),
    name="yolo11l_best",
    exist_ok=True,
)

test_model = YOLO(str(BEST_MODEL))

test_results = test_model.val(
    data=str(DATA_YAML_FIXED),
    split="test",
    device=DEVICE,
    plots=True,
    project=str(TEST_DIR),
    name="yolo11l_test",
    exist_ok=True,
)
