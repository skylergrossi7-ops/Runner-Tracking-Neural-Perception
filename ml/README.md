# Runner Perception Neural Network — Phase 1

Phase 1 builds a traceable, leakage-safe YOLO dataset and connects it to a
reproducible robust-training workflow. Reliable labels and independent
validation sessions remain the gate before any checkpoint is treated as a
meaningful result.

## Dataset contract

Each rosbag recording is one immutable session. Its images and labels share the
same session directory:

```text
data/runner_raw/
├── images/<session_id>/*.jpg
├── labels/<session_id>/*.txt
└── session_metadata.json
```

Never split adjacent frames from one recording across train and validation.
Record at least ten short, independent sessions spanning:

- distance: `near`, `mid`, `far`, and `mixed`
- lighting: `day`, `overcast`, `backlit`, and `lowlight`
- occlusion: `clear`, `partial`, and `occluded`
- environment: `simulation` and `real`

The current public-runner dataset contains 412 images from seven independent
video sessions, split into 360 training and 52 validation images without
session leakage. Raw Gazebo bags remain useful pipeline fixtures, but simulated
frames are not part of this real-world training set.

## 1. Extract one recording session

Run extraction with the isolated system ROS 2 Python environment so
`rosbag2_py` and `cv_bridge` use their compatible NumPy 1.x build. This avoids
the NumPy 2.x package currently installed in the ArduPilot virtual environment:

```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/Runner-Tracking-Neural-Perception

env -u VIRTUAL_ENV \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages \
  /usr/bin/python3 ml/scripts/extract_rosbag_images.py \
  --bag artifacts/loopback_bag_verification_20260823/camera_bag_gazebo \
  --dataset-root data/runner_raw \
  --session-id gazebo_runway_001 \
  --sample-hz 3 \
  --condition distance=mixed \
  --condition lighting=day \
  --condition occlusion=clear \
  --condition environment=simulation \
  --quality-flag runner_top_clipped \
  --quality-flag pipeline_smoke_test_only
```

Use a new `--session-id` for every rosbag. The extractor stores the source-bag
SHA-256 digest, topic, timestamps, conditions, and sampling rate in
`session_metadata.json`.

### Batch-ingest new recording sessions

Place each new ROS 2 recording in its own folder below an inbox directory. A
bag may contain one or more `.mcap` or `.db3` chunks. Create a tag file such as:

```json
{
  "sessions": {
    "outdoor_run_01": {
      "distance": "mid",
      "lighting": "normal",
      "occlusion": "none"
    },
    "outdoor_run_02": {
      "distance": "far",
      "lighting": "backlit",
      "occlusion": "partial"
    }
  }
}
```

Run unattended batch ingestion with the same isolated ROS Python environment:

```bash
source /opt/ros/jazzy/setup.bash
cd /path/to/Runner-Tracking-Neural-Perception

env -u VIRTUAL_ENV \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages \
  /usr/bin/python3 ml/scripts/ingest_sessions.py \
  --input-dir data/rosbag_inbox \
  --dataset-root data/runner_raw \
  --metadata-file data/rosbag_inbox/session_tags.json \
  --every-nth-frame 5 \
  --non-interactive
```

Omit `--metadata-file` and `--non-interactive` to answer controlled prompts for
each untagged bag. Sessions are assigned collision-free names such as `bag_001`,
duplicate source bags are skipped by SHA-256 hash, and every output is checked
for contiguous filenames, non-empty files, and successful image decoding. The
batch result is saved as `runner_raw/ingestion_report.json`.

## 2. Generate provisional labels

The pretrained COCO `person` class can accelerate annotation. These are teacher
suggestions, not ground truth:

```bash
cd /path/to/Runner-Tracking-Neural-Perception

python3 ml/scripts/preannotate_yolo.py \
  --dataset-root data/runner_raw \
  --session gazebo_runway_001 \
  --model yolov8n.pt \
  --source-class 0 \
  --target-class 0 \
  --confidence 0.15 \
  --review-threshold 0.35
```

`annotation_review.json` prioritizes missing, low-confidence, and multi-person
frames. Review every image in CVAT or another YOLO-compatible annotation tool.
Correct box edges, identify the intended runner when several people are visible,
and keep an empty `.txt` file for a verified negative image.

## 3. Validate and attest human review

After manual review is complete:

```bash
python3 ml/scripts/validate_yolo_dataset.py \
  --dataset-root data/runner_raw \
  --class-count 1 \
  --mark-reviewed
```

This checks image decoding, mirrored labels, class IDs, normalized box geometry,
duplicate images, metadata coverage, and annotation completeness. The
`--mark-reviewed` flag is rejected if validation fails or unlabeled images are
allowed.

## 4. Create runner_v1

After at least two independent sessions are reviewed (ten or more recommended):

```bash
python3 ml/scripts/prepare_dataset.py \
  --dataset-root data/runner_raw \
  --session-metadata data/runner_raw/session_metadata.json \
  --output-dir data/runner_v1 \
  --val-fraction 0.20 \
  --seed 42 \
  --path-mode relative
```

The output contains `train.txt`, `val.txt`, `splits_v1.json`, and
`runner_v1.yaml`. The split is deterministic, keeps full sessions exclusive,
and balances represented distance, lighting, and occlusion conditions when the
available sessions permit it.

## Source licensed public runner data

`source_public_runner_data.py` supports a local media directory, a selected
Roboflow Universe dataset version, or a selected Kaggle dataset. It scans YOLO
annotations when available, samples video at a fixed interval, and verifies
each proposed runner with a pretrained COCO-keypoint network. The strict filter
requires a visible head, shoulders, hips, knees, and ankles plus leg geometry
consistent with running. It rejects clipped people, standing poses, unrelated
classes, annotation/pose mismatches, and exact duplicate frames.

Install the provider tools only in the isolated ML environment:

```bash
python -m pip install -r ml/requirements-public-data.txt
```

Process locally downloaded, licensed videos or a YOLO export:

```bash
python ml/scripts/source_public_runner_data.py \
  --provider local \
  --input data/public_runner_clips \
  --output data/public_runner_v1
```

Download one reviewed Roboflow Universe version. The identifier is taken from
the Universe URL, and the API key is read from the environment rather than a
command-line argument or committed file:

```bash
export ROBOFLOW_API_KEY="YOUR_PRIVATE_KEY"

python ml/scripts/source_public_runner_data.py \
  --provider roboflow \
  --roboflow-version workspace/project/1 \
  --source-url "https://universe.roboflow.com/workspace/project/dataset/1" \
  --license-name "LICENSE_FROM_DATASET_PAGE" \
  --license-url "URL_TO_LICENSE_TERMS" \
  --accept-license \
  --output data/public_runner_v1
```

Or use the authenticated official Kaggle CLI with a reviewed dataset slug:

```bash
python ml/scripts/source_public_runner_data.py \
  --provider kaggle \
  --kaggle-dataset owner/dataset-name \
  --source-url "https://www.kaggle.com/datasets/owner/dataset-name" \
  --license-name "LICENSE_FROM_KAGGLE_PAGE" \
  --license-url "URL_TO_LICENSE_TERMS" \
  --accept-license \
  --output data/public_runner_v1
```

The generated directory uses standard Ultralytics paths:

```text
data/public_runner_v1/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
├── runner_v1.yaml
├── splits_v1.json
└── public_source_report.json
```

Whole videos or source directories remain exclusive to one split. A run with
only one independent session is rejected by default; the explicit
`--allow-single-session-frame-split` option exists for pipeline smoke tests but
must not be used for final evaluation. The neural gait filter is deliberately
conservative, but its output is still marked
`provisional_requires_human_review`: inspect every retained box before merging
this public subset into the deployment-camera dataset.

## Phase 1 exit criteria

- At least ten independent sessions, including real and simulated footage.
- Every frame has a reviewed runner label or an explicitly empty negative label.
- Dataset validation passes with no errors.
- Train and validation session sets are disjoint.
- `splits_v1.json`, `runner_v1.yaml`, and `dataset_report.json` are versioned.
- Raw images, weights, and training runs remain outside Git.

Once these criteria pass, Phase 2 is YOLO fine-tuning, evaluation by distance and
occlusion slice, and OpenVINO export with a measured latency budget.

## Phase 2 training entry point

Training uses a separate ML environment so Albumentations or PyTorch upgrades do
not reintroduce the NumPy conflict with ROS cv_bridge:

~~~bash
cd /path/to/Runner-Tracking-Neural-Perception
python3 -m venv "$HOME/venvs/runner-ml"
source "$HOME/venvs/runner-ml/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python ml/train.py --dry-run
python ml/train.py
~~~

The default profile is ml/config/train_robust.yaml. It fine-tunes the local
YOLOv8n checkpoint with outdoor color/geometric augmentation, Mosaic, MixUp,
CutMix, motion and Gaussian blur, coarse dropout, and wireless compression. Use
--model yolo11n.pt to select YOLO11n when its weights are available.

The trainer refuses unreviewed annotations or session leakage. After training,
it evaluates the selected best checkpoint independently on both the train and
validation splits, prints their metrics side-by-side, and writes
overfitting_report.json beside the saved checkpoints.

## Automated Gazebo data pipeline

The simulation-data orchestrator records ten independent camera sessions across
near, middle, and far runner distances; normal, backlit, and low lighting; and
clear or partially occluded paths. It then runs ingestion, session-count checks,
teacher pre-annotation, structural validation, session-exclusive splitting, and
a final split-readiness check:

~~~bash
cd /path/to/Runner-Tracking-Neural-Perception

unset PYTHONPATH VIRTUAL_ENV
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

# Inspect all commands without starting Gazebo.
/usr/bin/python3 ml/run_simulation_data_pipeline.py --dry-run

# Run the complete ten-session collection and processing matrix.
/usr/bin/python3 ml/run_simulation_data_pipeline.py
~~~

Use --scenario NAME, --limit N, or --duration SECONDS for a smaller collection.
Use --record-only or --process-only to resume one side of the workflow. The
default scenario matrix is ml/config/simulation_data_scenarios.yaml.

Every execution writes pipeline_execution_report.json below a timestamped
artifacts/simulation_data_pipeline directory. Simulation pre-annotations remain
marked as requiring human review; the training gate will not treat them as final
ground truth until the operator reviews and attests them.
