# Public Runner Dataset Card — v1

## Intended use

Single-class (`runner`) object detection for a camera-equipped autonomous drone.
The dataset is an initial development set for detector training, benchmarking,
annotation tooling, and ROS 2 integration. It is not a safety certification
dataset.

## Composition

| Split | Images | Runner boxes | Sessions |
| --- | ---: | ---: | ---: |
| Train | 360 | 384 | 5 |
| Validation | 52 | 52 | 2 |
| Total | 412 | 436 | 7 |

The sessions cover aerial parks, desert roads, misty forests, forest trails,
rural paths, road silhouettes, and open-field runners. Whole recording
sessions—not individual frames—are assigned to one split, preventing temporal
frame leakage.

## Sources and licensing

The frames were extracted from reviewed Pexels videos. Source identifiers and
the acknowledged license URL are recorded in `ml/manifests/` and
`data/public_runner_v1/public_source_report.json`. Images and videos are omitted
from Git so downstream users must retrieve them from their original sources and
accept the source license themselves.

## Annotation and quality control

Candidate boxes were generated with a pretrained person/pose model, filtered by
visible-keypoint and temporal running-gait checks, then audited using contact
sheets and independent detector proposals. The latest cleanup:

- removed overlapping duplicate-person boxes;
- removed boxes on shadows, rocks, vegetation, and dark non-human figures;
- replaced 18 incorrect desert-terrain boxes with boxes on the true runner;
- preserved original changed labels and a machine-readable cleanup log.

The final validator found 412 decodable labelled images, 436 valid normalized
boxes, no cross-image duplicates, and no warnings or errors.

## Known limitations

- The dataset is small for production deployment.
- Public-video domains do not reproduce every real drone altitude, lens,
  wireless-compression artifact, weather condition, or demographic.
- Several group-running frames contain people other than the selected target.
  The present semantic contract is target-runner detection, not exhaustive
  annotation of every visible person.
- Metric depth, target identity persistence, and closed-loop flight safety must
  be evaluated separately.

Future versions should add independent real drone-camera sessions, hard
negatives, stronger occlusion coverage, and held-out geographic domains.
