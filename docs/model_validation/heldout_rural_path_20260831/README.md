# Held-out rural-path video test — 2026-08-31

This test evaluates the first 75-epoch YOLO11n runner checkpoint on a complete
recording session that was assigned exclusively to validation by
`data/public_runner_v1/splits_v1.json`. No frame from the `rural_path` session
was used for model training.

## Method

- Source: Pexels rural-path aerial runner recording listed in the committed
  public-source manifest and governed by the Pexels License.
- Evaluation interval: first 5.005 seconds (300 frames at 59.94 FPS).
- Input resolution: 2560 x 1440; YOLO inference size: 640.
- Confidence threshold: 0.25; NMS IoU threshold: 0.50.
- Ground-truth check: all 34 curated YOLO labels occurring in the interval.
- Runtime benchmark: CPU-only WSL evaluation. Encoding and annotation are
  included in effective throughput, so this number is not a GPU flight-rate
  benchmark.

## Result

- Runner detections: 300 / 300 frames (100%).
- Longest detection gap: 0 frames.
- Mean confidence: 0.769 (minimum 0.398, maximum 0.945).
- Ground-truth matches at IoU >= 0.50: 34 / 34 (100%).
- Mean best IoU on labeled frames: 0.858.
- Mean CPU inference latency: 93.24 ms; p95: 134.01 ms.
- End-to-end CPU processing rate: 5.99 FPS, including drawing and encoding.

The accompanying video is H.264/yuv420p with fast-start metadata for browser
and Windows compatibility. The evaluator also overrides legacy one-class
checkpoint metadata so the semantic label appears as `runner`, not `item`.

## Artifacts

- `rural_path_heldout_annotated.mp4`: annotated five-second review video.
- `rural_path_heldout_preview.jpg`: representative frame.
- `metrics.json`: machine-readable measurements.

This focused test demonstrates temporal persistence on one held-out recording.
It does not replace multi-session testing, hard-negative testing, identity
tracking evaluation, or hardware-in-the-loop flight validation.
