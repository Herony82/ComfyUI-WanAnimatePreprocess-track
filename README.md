# ComfyUI-WanAnimatePreprocess-track

Custom node for ComfyUI that enhances `PoseAndFaceDetection` from [ComfyUI-WanAnimatePreprocess](https://github.com/kijai/ComfyUI-WanAnimatePreprocess) with manual point tracking, stable face crops and temporal filtering.

## Requirements

[ComfyUI-WanAnimatePreprocess](https://github.com/kijai/ComfyUI-WanAnimatePreprocess) must be installed in `custom_nodes/`.

## Node: Tracked Pose and Face Detection

Category: `WanAnimatePreprocess`

### Problem solved

On nearly-static subjects with shadows, lighting changes or transient occlusions (leaves, objects passing in front), the internal YOLO tracker can lock onto a shadow instead of the real face — making the generated rig unstable.

### How it works

1. **Run once** → the node saves the first frame and displays it in an interactive canvas inside the node UI
2. **Shift+click** on stable landmarks: pupils, nose, shoulders, hands, etc.
3. **Run again** → the node tracks those exact points with Lucas-Kanade optical flow across all frames, constrains YOLO+ViTPose to a ROI derived from those points, and produces stable face crops

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `roi_padding` | 1.3 | ROI expansion factor around tracked points bbox. Increase if ViTPose cuts off body parts. |
| `smooth_keypoints` | 0.0 | EMA smoothing on ViTPose keypoints (0=off, 1=maximum). Confidence-aware: undetected keypoints hold their last value instead of pulling the rig toward zero. |
| `face_crop_scale` | 1.5 | Face crop size multiplier relative to tracked point spread. Increase if face is cut off at edges. |
| `face_median_window` | 5 | Temporal median window (frames) on face crop pixels. Removes transient occlusions (leaves, shadows). 1=disabled. |

### Canvas controls

- **Shift+click** — add tracking point
- **Right-click** on existing point — remove it
- **Undo** — remove last point
- **Clear** — remove all points
- Points are saved in the workflow JSON and restored on reload

### Outputs

Drop-in compatible with `DrawViTPose`, `FaceBBoxToMask`, `WanVideoAnimateEmbeds` — identical output types to the original `PoseAndFaceDetection`.

| Output | Description |
|---|---|
| `pose_data` | ViTPose rig (smoothed if smooth_keypoints > 0) |
| `face_images` | Stable face crops — fixed bbox centered on tracked point median centroid + temporal median |
| `key_frame_body_points` | JSON keyframe data |
| `bboxes` | Person detection bbox |
| `face_bboxes` | Fixed face bboxes (same stability as face_images) |

## Installation

```
ComfyUI/custom_nodes/
├── ComfyUI-WanAnimatePreprocess/    ← required dependency
└── ComfyUI-WanAnimatePreprocess-track/  ← this package
```

Clone or extract this repo into `ComfyUI/custom_nodes/` and restart ComfyUI.
