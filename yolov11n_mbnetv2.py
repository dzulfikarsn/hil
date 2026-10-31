from pathlib import Path
from datetime import datetime
import json
import yaml
import torch
import ultralytics
import pandas as pd


DATASET_NAME = "LeukemiaAttri_YOLO"
MODEL_NAME = "yolo11n_mobilenetv2"

DATA_INPUT = Path("/home/esr/.cache/kagglehub/datasets/albatani/leukemiaattri-yolo-train-val-test/versions/1")

OUTPUT_ROOT = Path(
    "/home/esr/isim/hil/hil3"
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
    for path in DATA_INPUT.rglob(DATASET_NAME)
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
        f"di {DATA_INPUT}."
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
print("Ready for YOLO11n MobileNetV2 training.")

EXPECTED_CLASS_NAMES = [
    "None", "Myeloblast", "Lymphoblast", "Neutrophil",
    "Atypical lymphocyte", "Promonocyte", "Monoblast", "Lymphocyte",
    "Myelocyte", "Abnormal promyelocyte", "Monocyte",
    "Metamyelocyte", "Eosinophil", "Basophil"
]
EXPECTED_NUM_CLASSES = len(EXPECTED_CLASS_NAMES)

from pathlib import Path
import torch
from ultralytics import YOLO

# Gunakan konfigurasi dataset dan direktori yang sudah disiapkan pada Cell 2.
NC = EXPECTED_NUM_CLASSES
if NC != num_classes:
    raise ValueError(
        f"Jumlah kelas tidak konsisten: EXPECTED_NUM_CLASSES={NC}, data.yaml={num_classes}"
    )

MODEL_YAML = SUMMARY_DIR / "yolo11n_mobilenetv2.yaml"
NECK_WIDTH = 0.25
P5_ARG = int(round(320 / NECK_WIDTH))  # 1280 sebelum scaling oleh Ultralytics

MODEL_YAML_TEXT = f"""
nc: {NC}
scale: n
scales:
  n: [0.50, {NECK_WIDTH}, 1024]

backbone:
  - [-1, 1, TorchVision, [320, mobilenet_v2, DEFAULT, True, 2, True]]
  # MobileNetV2 feature maps: stride 8 (32 ch), stride 16 (96 ch), stride 32 (1280 ch).
  - [0, 1, Index, [32, 6]]
  - [0, 1, Index, [96, 13]]
  - [0, 1, Index, [320, 18]]
  - [-1, 1, SPPF, [{P5_ARG}, 5]]
  - [-1, 2, C2PSA, [{P5_ARG}]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 1], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[11, 14, 17], 1, Detect, [nc]]
"""

MODEL_YAML.write_text(MODEL_YAML_TEXT, encoding="utf-8")
print("Model YAML:", MODEL_YAML)

architecture_check = YOLO(str(MODEL_YAML))
architecture_check.info(detailed=False)
architecture_check.model.eval()

with torch.no_grad():
    _ = architecture_check.model(torch.zeros(1, 3, 640, 640))

stride_values = [
    float(value)
    for value in architecture_check.model.stride.detach().cpu().tolist()
]
detect_head = architecture_check.model.model[-1]
model_nc = getattr(detect_head, "nc", None)

print("Stride model:", stride_values)
print("Jumlah kelas pada Detect head:", model_nc)

if stride_values != [8.0, 16.0, 32.0]:
    raise ValueError(f"Stride model tidak sesuai: {stride_values}")

if model_nc != NC:
    raise ValueError(
        f"Jumlah kelas Detect head ({model_nc}) tidak sama dengan dataset ({NC})"
    )

del architecture_check

model = YOLO(str(MODEL_YAML))
results = model.train(
    data=str(DATA_YAML_FIXED),
    epochs=150,
    project=str(TRAIN_DIR),
    name="yolo11n_mobilenetv2",
    device=DEVICE,
    plots=True,
    save=True,
    exist_ok=True,
    verbose=True,
)

TRAIN_RUN_DIR = Path(model.trainer.save_dir)
WEIGHTS_DIR = TRAIN_RUN_DIR / "weights"
BEST_MODEL = WEIGHTS_DIR / "best.pt"
LAST_MODEL = WEIGHTS_DIR / "last.pt"

if not BEST_MODEL.is_file():
    raise FileNotFoundError(
        f"Bobot best.pt tidak ditemukan setelah training: {BEST_MODEL}"
    )

print("\\nTRAINING SELESAI")
print("Model YAML       :", MODEL_YAML)
print("Training run     :", TRAIN_RUN_DIR)
print("Best weights     :", BEST_MODEL)
print("Last weights     :", LAST_MODEL)

from pathlib import Path

TRAIN_RUN_DIR = Path(TRAIN_RUN_DIR)
BEST_MODEL = Path(BEST_MODEL)
LAST_MODEL = Path(LAST_MODEL)

if not BEST_MODEL.is_file():
    raise FileNotFoundError(
        "Bobot terbaik model modifikasi tidak ditemukan"
    )

print("MODEL TRAINING YANG AKAN DIEVALUASI")
print("Training directory:", TRAIN_RUN_DIR)
print("Best model        :", BEST_MODEL)
print("Last model        :", LAST_MODEL)
print("Dataset YAML      :", DATA_YAML_FIXED)

# Muat best.pt hasil training yang sama untuk proses validasi dan testing.
best_model = YOLO(str(BEST_MODEL))
print("Best checkpoint berhasil dimuat.")

EVALUATION_MODEL = YOLO(str(BEST_MODEL))

val_results = EVALUATION_MODEL.val(
    data=str(DATA_YAML_FIXED),
    split="val",
    device=DEVICE,
    plots=True,
    project=str(VAL_DIR),
    name="yolo11n_mobilenetv2_best",
    exist_ok=True,
)

test_results = EVALUATION_MODEL.val(
    data=str(DATA_YAML_FIXED),
    split="test",
    device=DEVICE,
    plots=True,
    project=str(TEST_DIR),
    name="yolo11n_mobilenetv2_test",
    exist_ok=True,
)
