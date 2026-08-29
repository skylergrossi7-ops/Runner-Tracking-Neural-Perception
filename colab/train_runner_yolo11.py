"""Self-contained Google Colab trainer for the runner YOLO dataset.

Paste this file into one Colab cell, or upload and execute it with `%run`.
Set DATASET_URL to a direct/Google Drive URL, or leave it blank for upload.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


DATASET_URL = ""
MODEL = "yolo11n.pt"
EPOCHS = 75
IMAGE_SIZE = 640
RUN_NAME = "runner_yolo11n_75epochs"
DOWNLOAD_RESULTS = True

WORKSPACE = Path("/content/runner_training")
ARCHIVE = WORKSPACE / "public_runner_v1.zip"
EXTRACTED = WORKSPACE / "dataset"
RUNS = WORKSPACE / "runs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def install_dependencies():
    """Install the training dependencies into the active Colab runtime."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "ultralytics>=8.3,<9",
            "PyYAML",
            "gdown",
            "requests",
        ],
        check=True,
    )


def acquire_archive():
    """Download the configured archive or ask the Colab user to upload one."""
    if DATASET_URL.strip():
        if "drive.google.com" in DATASET_URL:
            import gdown

            result = gdown.download(
                url=DATASET_URL, output=str(ARCHIVE), fuzzy=True, quiet=False
            )
            if not result:
                raise RuntimeError("Google Drive dataset download failed")
            return

        import requests

        with requests.get(DATASET_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with ARCHIVE.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        output.write(chunk)
        return

    from google.colab import files

    print("Upload the complete public_runner_v1 ZIP archive.")
    uploaded = files.upload()
    archives = [name for name in uploaded if name.lower().endswith(".zip")]
    if len(archives) != 1:
        raise RuntimeError("Upload exactly one ZIP archive")
    shutil.move(archives[0], ARCHIVE)


def safe_extract(archive, destination):
    """Extract an archive while rejecting directory traversal entries."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            output = (destination / member.filename).resolve()
            if os.path.commonpath([str(destination), str(output)]) != str(
                destination
            ):
                raise RuntimeError(f"Unsafe ZIP entry: {member.filename}")
        source.extractall(destination)


def locate_dataset():
    """Return the directory containing images/train and labels/train."""
    for train_images in EXTRACTED.rglob("images/train"):
        root = train_images.parent.parent
        required = [
            root / "images/train",
            root / "images/val",
            root / "labels/train",
            root / "labels/val",
        ]
        if all(path.is_dir() for path in required):
            return root.resolve()
    raise FileNotFoundError("The ZIP has no standard YOLO train/val layout")


def validate_split(root, split):
    """Check that every image has one syntactically valid YOLO label file."""
    image_root = root / "images" / split
    label_root = root / "labels" / split
    images = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No {split} images found")
    objects = 0
    for image in images:
        label = label_root / image.relative_to(image_root).with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing label for {image}")
        for number, line in enumerate(label.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label}:{number}: expected five fields")
            class_id = int(float(fields[0]))
            x, y, width, height = map(float, fields[1:])
            if class_id != 0 or not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise ValueError(f"{label}:{number}: invalid YOLO box")
            objects += 1
    return {"images": len(images), "objects": objects}


def main():
    """Prepare the dataset, train for 75 epochs, validate and download."""
    install_dependencies()
    import torch
    import yaml
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Select Runtime > Change runtime type > T4 GPU."
        )
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    EXTRACTED.mkdir(parents=True)
    RUNS.mkdir(parents=True)
    acquire_archive()
    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError("The selected dataset is not a valid ZIP archive")
    safe_extract(ARCHIVE, EXTRACTED)
    dataset = locate_dataset()
    summary = {
        split: validate_split(dataset, split) for split in ("train", "val")
    }
    print("Dataset validation passed:", json.dumps(summary, indent=2))

    data_yaml = WORKSPACE / "runner_v1_colab.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": {0: "runner"},
            },
            sort_keys=False,
        )
    )

    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=-1,
        device=0,
        workers=2,
        project=str(RUNS),
        name=RUN_NAME,
        exist_ok=True,
        single_cls=True,
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        patience=EPOCHS,
        seed=42,
        deterministic=True,
        amp=True,
        cache=False,
        save=True,
        save_period=10,
        plots=True,
        val=True,
        hsv_h=0.015,
        hsv_s=0.45,
        hsv_v=0.35,
        degrees=5.0,
        translate=0.12,
        scale=0.45,
        shear=2.0,
        perspective=0.0003,
        fliplr=0.5,
        mosaic=0.75,
        mixup=0.08,
        close_mosaic=10,
    )

    run_directory = Path(model.trainer.save_dir)
    best = run_directory / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError("Training finished without weights/best.pt")
    metrics = YOLO(str(best)).val(
        data=str(data_yaml), imgsz=IMAGE_SIZE, device=0, workers=2
    )
    report = {
        "dataset": summary,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    (run_directory / "final_metrics.json").write_text(
        json.dumps(report, indent=2)
    )
    archive = shutil.make_archive(
        str(WORKSPACE / RUN_NAME), "zip", root_dir=run_directory
    )
    print("Training complete:", json.dumps(report, indent=2))
    if DOWNLOAD_RESULTS:
        from google.colab import files

        files.download(archive)


if __name__ == "__main__":
    main()
