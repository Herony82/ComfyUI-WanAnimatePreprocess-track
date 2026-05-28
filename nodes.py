"""
nodes.py  -  TrackedPoseAndFaceDetection
=========================================
Part of ComfyUI-WanAnimatePreprocess-track.

Pipeline:
  1. Prima esecuzione (nessun punto): detection normale, salva primo frame
     nel canvas JS per la selezione dei punti.
  2. Con punti selezionati:
     - Optical flow (Lucas-Kanade) traccia i punti su tutti i frame
     - ROI calcolata dal bounding box di TUTTI i punti (faccia + corpo + mani)
     - Ogni frame viene mascherato fuori dalla ROI prima di YOLO+ViTPose
     - YOLO non può piu' "sbandare" su ombre o oggetti esterni
     - Face crop estratte dall'originale non mascherato
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys

import cv2
import numpy as np
import torch
import comfy.utils
import folder_paths
from PIL import Image as PILImage


# ─────────────────────────────────────────────────────────────────────────────
#  Import dal pacchetto padre ComfyUI-WanAnimatePreprocess
# ─────────────────────────────────────────────────────────────────────────────

def _import_wan(module_name: str):
    candidates = [
        f"custom_nodes.ComfyUI-WanAnimatePreprocess.{module_name}",
        f"ComfyUI-WanAnimatePreprocess.{module_name}",
        f"comfyui_wananimatepreprocess.{module_name}",
    ]
    for key in candidates:
        if key in sys.modules:
            return sys.modules[key]
    for key in candidates:
        try:
            return importlib.import_module(key)
        except ModuleNotFoundError:
            pass
    try:
        for p in folder_paths.get_folder_paths("custom_nodes"):
            d = os.path.join(p, "ComfyUI-WanAnimatePreprocess")
            if os.path.isdir(d):
                parent = os.path.dirname(d)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                break
        return importlib.import_module(
            f"ComfyUI-WanAnimatePreprocess.{module_name}"
        )
    except Exception:
        pass
    raise ImportError(
        f"[TrackedPoseAndFaceDetection] Impossibile importare "
        f"'ComfyUI-WanAnimatePreprocess.{module_name}'. "
        "Assicurati che ComfyUI-WanAnimatePreprocess sia installato in custom_nodes."
    )


_PoseNode = _import_wan("nodes").PoseAndFaceDetection


# ─────────────────────────────────────────────────────────────────────────────
#  Optical flow
# ─────────────────────────────────────────────────────────────────────────────

def track_optical_flow(
    frames_uint8: np.ndarray,
    seed_points: list,
) -> np.ndarray:
    """
    Traccia seed_points su tutti i frame con Lucas-Kanade.

    frames_uint8 : [N, H, W, 3]  uint8
    seed_points  : list of [x, y]  (pixel coords su frame 0)

    Restituisce tracked [N, P, 2]  float32.
    Se un punto viene perso, mantiene l'ultima posizione valida.
    """
    N, H, W, _ = frames_uint8.shape
    P = len(seed_points)

    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_uint8]

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001),
    )

    pts     = np.array(seed_points, dtype=np.float32).reshape(-1, 1, 2)
    tracked = np.zeros((N, P, 2), dtype=np.float32)
    tracked[0] = pts.reshape(P, 2)

    cur = pts.copy()

    for f in range(1, N):
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[f - 1], gray[f], cur, None, **lk_params
        )
        good = status.reshape(-1) == 1
        for i in range(P):
            if good[i]:
                cur[i] = nxt[i]
            # punto perso: mantieni posizione precedente (cur[i] invariato)
        tracked[f] = cur.reshape(P, 2)

    return tracked  # [N, P, 2]


# ─────────────────────────────────────────────────────────────────────────────
#  ROI da punti trackati
# ─────────────────────────────────────────────────────────────────────────────

def compute_roi(
    points_2d: np.ndarray,
    frame_w: int,
    frame_h: int,
    padding: float = 1.3,
) -> tuple:
    """
    Bounding box di tutti i punti trackati con padding.

    points_2d : [P, 2]  coordinate pixel per un singolo frame
    padding   : moltiplicatore di espansione (1.3 = +30% in ogni direzione)

    Restituisce (x1, y1, x2, y2) clampati ai bordi del frame.
    """
    pts = np.asarray(points_2d)
    if len(pts) == 0:
        return 0, 0, frame_w, frame_h

    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    hw = (x_max - x_min) / 2.0 * padding
    hh = (y_max - y_min) / 2.0 * padding

    # Minimo 32px per evitare ROI degeneri
    hw = max(hw, 32.0)
    hh = max(hh, 32.0)

    x1 = max(0,       int(cx - hw))
    y1 = max(0,       int(cy - hh))
    x2 = min(frame_w, int(cx + hw))
    y2 = min(frame_h, int(cy + hh))

    return x1, y1, x2, y2


def apply_roi_masks(
    frames_uint8: np.ndarray,
    tracked: np.ndarray,
    padding: float,
) -> np.ndarray:
    """
    Per ogni frame: scurisce fortemente tutto fuori dalla ROI (gamma 0.08)
    invece di azzerare completamente.

    Questo permette a YOLO di preferire il soggetto dentro la ROI (molto
    più luminoso) senza impedire a ViTPose di vedere il resto del corpo.

    frames_uint8 : [N, H, W, 3]  uint8
    tracked      : [N, P, 2]

    Restituisce masked [N, H, W, 3]  uint8.
    """
    N, H, W, _ = frames_uint8.shape
    out = frames_uint8.copy()

    # LUT gamma per scurire l'esterno (0.08 = quasi nero ma non zero)
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / 0.08)
    lut = np.clip(lut * 255, 0, 255).astype(np.uint8)

    for i in range(N):
        x1, y1, x2, y2 = compute_roi(tracked[i], W, H, padding)
        # Salva la ROI originale
        roi_orig = frames_uint8[i, y1:y2, x1:x2].copy()
        # Scurisci tutto il frame
        out[i] = cv2.LUT(frames_uint8[i], lut)
        # Ripristina la ROI originale (piena luminosità)
        out[i, y1:y2, x1:x2] = roi_orig

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Face crop — Opzione 4: bbox dai tracked points + temporal median
# ─────────────────────────────────────────────────────────────────────────────

def _tracked_face_bbox(
    tracked_frame: np.ndarray,
    frame_w: int,
    frame_h: int,
    crop_scale: float,
    fixed_size_px: int,
) -> tuple:
    """
    Calcola il bbox della face crop centrato sulla centroide dei punti
    trackati, con dimensione FISSA (fixed_size_px × fixed_size_px).

    tracked_frame : [P, 2]  coordinate pixel
    crop_scale    : moltiplicatore rispetto alla spread dei punti
                    (usato solo per calcolare la dimensione se fixed_size_px=0)
    fixed_size_px : lato del quadrato in pixel sul frame originale.
                    Se 0, viene calcolato dalla spread dei punti.

    Restituisce (x1, y1, x2, y2) clampati ai bordi.
    """
    pts = np.asarray(tracked_frame)
    cx  = float(pts[:, 0].mean())
    cy  = float(pts[:, 1].mean())

    if fixed_size_px > 0:
        half = fixed_size_px / 2.0
    else:
        # Fallback: usa la spread dei punti con crop_scale
        spread = max(
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            10.0,
        )
        half = spread / 2.0 * crop_scale

    x1 = max(0,       int(cx - half))
    y1 = max(0,       int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half))

    # Se clampato ai bordi, aggiusta l'altro lato per mantenere la dimensione
    if x2 - x1 < int(half * 2):
        if x1 == 0:
            x2 = min(frame_w, int(half * 2))
        else:
            x1 = max(0, x2 - int(half * 2))
    if y2 - y1 < int(half * 2):
        if y1 == 0:
            y2 = min(frame_h, int(half * 2))
        else:
            y1 = max(0, y2 - int(half * 2))

    return x1, y1, x2, y2


def extract_face_crops_tracked_median(
    images: torch.Tensor,
    tracked: np.ndarray,
    face_median_window: int,
    face_size: int = 512,
    crop_scale: float = 1.5,
) -> torch.Tensor:
    """
    Opzione 4: face crop con bbox FISSO dai punti trackati + temporal median.

    1. Calcola la dimensione fissa dal primo frame (spread dei punti × crop_scale)
    2. Per ogni frame, centra il crop sulla centroide dei punti trackati
    3. Applica temporal median sui pixel (finestra face_median_window)
       per eliminare occlusioni transitorie (foglie, ombre improvvise)

    images             : [N, H, W, 3]  float32  [0, 1]
    tracked            : [N, P, 2]     pixel coords
    face_median_window : finestra mediana (frame). 1 = disabilitato.
    crop_scale         : moltiplicatore spread punti → dimensione crop

    Restituisce [N, face_size, face_size, 3]  float32  [0, 1].
    """
    N      = int(images.shape[0])
    orig_h = int(images.shape[1])
    orig_w = int(images.shape[2])
    imgs   = (images.cpu().numpy() * 255).astype(np.uint8)

    # ── Calcola dimensione fissa dalla spread dei punti sul frame 0 ──────────
    pts0    = np.asarray(tracked[0])
    spread0 = max(
        pts0[:, 0].max() - pts0[:, 0].min(),
        pts0[:, 1].max() - pts0[:, 1].min(),
        20.0,
    )
    fixed_px = int(spread0 * crop_scale)
    fixed_px = max(fixed_px, 64)   # minimo 64px
    print(f"[TrackedPose] Face crop: dimensione fissa {fixed_px}×{fixed_px}px "
          f"(spread={spread0:.1f}px × scale={crop_scale})")

    # ── Centro fisso: mediana del centroide su tutti i frame ─────────────────
    # Invece di usare tracked[i] per ogni frame (causa zoom in/out per drift
    # minimo del centroide), usiamo la mediana su tutti i frame come centro
    # costante. Su soggetto quasi statico il centroide non si sposta
    # significativamente — fissarlo elimina il micro-zoom.
    all_cx = np.median([tracked[i][:, 0].mean() for i in range(N)])
    all_cy = np.median([tracked[i][:, 1].mean() for i in range(N)])
    half   = fixed_px / 2.0

    x1_fixed = max(0,       int(all_cx - half))
    y1_fixed = max(0,       int(all_cy - half))
    x2_fixed = min(orig_w,  int(all_cx + half))
    y2_fixed = min(orig_h,  int(all_cy + half))

    # Aggiusta se clampato ai bordi
    if x2_fixed - x1_fixed < fixed_px and x1_fixed == 0:
        x2_fixed = min(orig_w, fixed_px)
    if y2_fixed - y1_fixed < fixed_px and y1_fixed == 0:
        y2_fixed = min(orig_h, fixed_px)

    print(f"[TrackedPose] Face crop centro fisso: "
          f"cx={all_cx:.1f} cy={all_cy:.1f} "
          f"bbox=({x1_fixed},{y1_fixed},{x2_fixed},{y2_fixed})")

    # ── Estrai crop raw per ogni frame (bbox identico per tutti i frame) ──────
    raw_crops = []
    bboxes    = []
    for i in range(N):
        bboxes.append((x1_fixed, y1_fixed, x2_fixed, y2_fixed))
        crop = imgs[i, y1_fixed:y2_fixed, x1_fixed:x2_fixed]
        if crop.size == 0:
            crop = imgs[i, orig_h//4:3*orig_h//4, orig_w//4:3*orig_w//4]
        raw_crops.append(cv2.resize(crop, (face_size, face_size)))

    raw_stack = np.stack(raw_crops, axis=0).astype(np.float32)  # [N, S, S, 3]

    # ── Temporal median per rimuovere occlusioni transitorie ─────────────────
    w = face_median_window
    if w > 1:
        w    = w if w % 2 == 1 else w + 1   # forza dispari
        half = w // 2
        smoothed = raw_stack.copy()
        for i in range(N):
            s = max(0, i - half)
            e = min(N, i + half + 1)
            smoothed[i] = np.median(raw_stack[s:e], axis=0)
        raw_stack = smoothed

    arr    = np.clip(raw_stack, 0, 255).astype(np.uint8)
    tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
    return tensor, bboxes


def extract_face_crops_from_bboxes(
    images: torch.Tensor,
    face_bboxes: list,
    face_size: int = 512,
) -> torch.Tensor:
    """
    Fallback (usato quando non ci sono tracked points):
    crop dal video originale usando i bbox del nodo base.
    """
    N      = int(images.shape[0])
    orig_h = int(images.shape[1])
    orig_w = int(images.shape[2])
    imgs   = (images.cpu().numpy() * 255).astype(np.uint8)
    crops  = []

    for i in range(N):
        idx = min(i, len(face_bboxes) - 1)
        bb  = face_bboxes[idx]
        if hasattr(bb, "__len__") and len(bb) >= 4:
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        else:
            x1, y1 = orig_w // 4, orig_h // 4
            x2, y2 = 3 * orig_w // 4, 3 * orig_h // 4
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(orig_w, x2);  y2 = min(orig_h, y2)
        crop = imgs[i, y1:y2, x1:x2]
        if crop.size == 0:
            crop = imgs[i, orig_h//4:3*orig_h//4, orig_w//4:3*orig_w//4]
        crops.append(cv2.resize(crop, (face_size, face_size)))

    arr = np.stack(crops).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal median su tensor face_imgs (fallback senza tracking)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_pixel_median(face_imgs: torch.Tensor, window: int) -> torch.Tensor:
    """
    Temporal median sui pixel di face_imgs [N, H, W, 3].
    Usato nel path senza tracking per rimuovere occlusioni transitorie
    anche quando non ci sono punti selezionati.
    """
    w = window if window % 2 == 1 else window + 1
    half = w // 2
    arr  = (face_imgs.cpu().numpy() * 255).astype(np.float32)
    N    = arr.shape[0]
    out  = arr.copy()
    for i in range(N):
        s = max(0, i - half)
        e = min(N, i + half + 1)
        out[i] = np.median(arr[s:e], axis=0)
    return torch.from_numpy(
        np.clip(out, 0, 255).astype(np.float32) / 255.0
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Salva primo frame per il canvas JS
# ─────────────────────────────────────────────────────────────────────────────

def save_first_frame(images: torch.Tensor) -> dict:
    """
    Salva il primo frame in temp/ e restituisce le info per il frontend.
    """
    first = images[0]
    arr   = (first.cpu().numpy() * 255).astype(np.uint8)
    pil   = PILImage.fromarray(arr)

    temp_dir = folder_paths.get_temp_directory()
    fname    = "tracked_pose_first_frame.png"
    fpath    = os.path.join(temp_dir, fname)
    pil.save(fpath)

    return {"filename": fname, "subfolder": "", "type": "temp"}


# ─────────────────────────────────────────────────────────────────────────────
#  Unpack risultato dal nodo base (struttura variabile tra versioni)
# ─────────────────────────────────────────────────────────────────────────────

_result_logged = False
_pose_struct_logged = False


def _log_pose_structure(pose_data):
    """Stampa la struttura di pose_data una volta sola per il debug."""
    global _pose_struct_logged
    if _pose_struct_logged:
        return
    _pose_struct_logged = True

    print("=" * 60)
    print("[TrackedPose] DEBUG pose_data structure:")
    print(f"  type(pose_data) = {type(pose_data)}")

    if isinstance(pose_data, dict):
        print(f"  keys = {list(pose_data.keys())}")
        for k, v in pose_data.items():
            print(f"    [{k}] type={type(v)} repr={repr(v)[:80]}")
            if isinstance(v, (list, tuple)) and len(v) > 0:
                first = v[0]
                print(f"      first item type={type(first)}")
                pd = getattr(first, "__dict__", None)
                if pd:
                    print(f"      first item __dict__ keys = {list(pd.keys())[:20]}")
                    for ak, av in list(pd.items())[:10]:
                        try:
                            arr = np.asarray(av)
                            print(f"        .{ak}: shape={arr.shape} dtype={arr.dtype} "
                                  f"sample={arr.flat[:3].tolist()}")
                        except Exception:
                            print(f"        .{ak}: {repr(av)[:60]}")
    elif isinstance(pose_data, (list, tuple)) and len(pose_data) > 0:
        print(f"  len = {len(pose_data)}")
        first = pose_data[0]
        print(f"  first item type={type(first)}")
        pd = getattr(first, "__dict__", None)
        if pd:
            print(f"  first item __dict__ keys = {list(pd.keys())[:20]}")
            for ak, av in list(pd.items())[:10]:
                try:
                    arr = np.asarray(av)
                    print(f"    .{ak}: shape={arr.shape} dtype={arr.dtype} "
                          f"sample={arr.flat[:3].tolist()}")
                except Exception:
                    print(f"    .{ak}: {repr(av)[:60]}")
    else:
        pd = getattr(pose_data, "__dict__", None)
        if pd:
            print(f"  __dict__ keys = {list(pd.keys())[:20]}")
    print("=" * 60)


def _unpack_base_result(raw):
    """
    Il nodo base può restituire una tupla diretta o un dict con 'result'.
    Restituisce sempre (pose_data, face_images, kfbp, bboxes, face_bboxes).
    """
    global _result_logged

    if isinstance(raw, dict) and "result" in raw:
        raw = raw["result"]

    if not _result_logged:
        print(
            f"[TrackedPose] base node result: type={type(raw)}, "
            f"len={len(raw) if hasattr(raw, '__len__') else '?'}"
        )
        _result_logged = True

    if isinstance(raw, (list, tuple)) and len(raw) >= 5:
        result = tuple(raw[:5])
        _log_pose_structure(result[0])   # logga pose_data
        return result

    raise RuntimeError(
        f"[TrackedPose] Il nodo base ha restituito un formato inatteso: "
        f"{type(raw)}. Controlla la versione di ComfyUI-WanAnimatePreprocess."
    )



# ─────────────────────────────────────────────────────────────────────────────
#  Smoothing keypoints ViTPose (EMA confidence-aware)
# ─────────────────────────────────────────────────────────────────────────────
#
#  Struttura reale di pose_data (da debug log):
#    pose_data = {
#        "pose_metas": [AAPoseMeta, ...],   <- lista per frame
#        "pose_metas_original": [...],
#        "retarget_image": None,
#        "refer_pose_meta": None,
#    }
#    AAPoseMeta.__dict__ = {
#        kps_body:   (20, 2) float64
#        kps_lhand:  (21, 2) float64
#        kps_rhand:  (21, 2) float64
#        kps_face:   (69, 2) float64
#        kps_body_p: (20,)   float32
#        kps_lhand_p:(21,)   float32
#        kps_rhand_p:(21,)   float32
#        kps_face_p: (69,)   float32
#    }

_CONF_THR = 0.15   # keypoint sotto questa confidence = non rilevato

_KPS_PAIRS = [
    ("kps_body",  "kps_body_p"),
    ("kps_lhand", "kps_lhand_p"),
    ("kps_rhand", "kps_rhand_p"),
    ("kps_face",  "kps_face_p"),
]


def _smooth_pose_data(pose_data: dict, alpha: float) -> dict:
    """
    EMA confidence-aware sui keypoints di tutti gli AAPoseMeta in pose_metas.

    alpha = 0.0  -> pass-through (nessuno smoothing)
    alpha = 1.0  -> massimo smoothing (rig quasi fermo)

    I keypoint con confidence < _CONF_THR mantengono il valore del frame
    precedente senza aggiornare lo stato EMA — non tirano il rig verso zero.
    """
    if alpha <= 0.0:
        return pose_data

    if not isinstance(pose_data, dict):
        print("[TrackedPose] smooth: pose_data non è un dict, skip.")
        return pose_data

    poses = pose_data.get("pose_metas")
    if not poses or len(poses) < 2:
        return pose_data

    import copy as _copy
    poses = [_copy.deepcopy(p) for p in poses]

    for kps_attr, conf_attr in _KPS_PAIRS:
        kps0 = getattr(poses[0], kps_attr, None)
        if kps0 is None:
            continue

        # stato EMA: inizializzato col frame 0
        prev = np.asarray(kps0, dtype=np.float64).copy()  # (K, 2)

        for p in poses[1:]:
            curr_raw = getattr(p, kps_attr, None)
            conf_raw = getattr(p, conf_attr, None)
            if curr_raw is None:
                continue

            curr = np.asarray(curr_raw, dtype=np.float64)   # (K, 2)
            conf = np.asarray(conf_raw, dtype=np.float64).ravel()                    if conf_raw is not None                    else np.ones(curr.shape[0], dtype=np.float64)

            K   = curr.shape[0]
            new = prev.copy()

            for k in range(K):
                if conf[k] >= _CONF_THR:
                    # keypoint rilevato: blend EMA e aggiorna stato
                    new[k] = alpha * prev[k] + (1.0 - alpha) * curr[k]
                    prev[k] = new[k]
                else:
                    # keypoint non rilevato: tieni precedente, NON aggiornare
                    new[k] = prev[k]

            # Riscrivi sul posto con dtype originale
            orig_dtype = np.asarray(curr_raw).dtype
            setattr(p, kps_attr, new.astype(orig_dtype))

    out = dict(pose_data)
    out["pose_metas"] = poses
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Nodo principale
# ─────────────────────────────────────────────────────────────────────────────

class TrackedPoseAndFaceDetection:
    """
    PoseAndFaceDetection con selezione manuale dei punti di tracking.

    Flusso d'uso
    ------------
    1. Collega il video a ``images`` ed esegui una volta.
       Il canvas nel nodo mostra il primo frame.
    2. Shift+click per aggiungere punti sui landmark stabili del soggetto:
       volto (pupille, naso), spalle, mani, fianchi, ecc.
       Click destro su un punto esistente per rimuoverlo.
    3. Esegui di nuovo. Il nodo traccia quei punti con optical flow,
       costruisce una ROI per frame e vincola YOLO+ViTPose a quell'area.

    Vantaggi rispetto all'originale
    --------------------------------
    - YOLO non può rilevare ombre o oggetti fuori dalla ROI definita dai
      punti dell'utente.
    - I punti coprono sia il volto che il corpo, quindi ViTPose riceve
      sempre il soggetto corretto indipendentemente dall'illuminazione.
    - Face crop estratte dal video originale non mascherato.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "POSEMODEL",
                    {"tooltip": "Da OnnxDetectionModelLoader (kijai)."},
                ),
                "images": (
                    "IMAGE",
                    {"tooltip": "Video originale. Esegui una volta per "
                                "caricare il primo frame nel canvas."},
                ),
                "width": (
                    "INT",
                    {"default": 832, "min": 64, "max": 2048, "step": 1},
                ),
                "height": (
                    "INT",
                    {"default": 480, "min": 64, "max": 2048, "step": 1},
                ),
                "tracking_points": (
                    "STRING",
                    {"default": "[]", "multiline": False,
                     "tooltip": "JSON gestito dal canvas. Non modificare manualmente."},
                ),
                "roi_padding": (
                    "FLOAT",
                    {
                        "default": 1.3,
                        "min": 1.05,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": (
                            "Espansione della ROI rispetto al bbox dei punti. "
                            "1.3 = +30% in ogni direzione. "
                            "Aumenta solo se il nodo base taglia parti del corpo."
                        ),
                    },
                ),
                "smooth_keypoints": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Smoothing EMA sui keypoints ViTPose. "
                            "0.0 = nessuno smoothing (raw). "
                            "0.5 = smoothing moderato (consigliato). "
                            "1.0 = massimo smoothing. "
                            "I keypoint a bassa confidence tengono il valore "
                            "precedente invece di tirare il rig verso zero."
                        ),
                    },
                ),
                "face_crop_scale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.5,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": (
                            "Moltiplicatore della dimensione della face crop "
                            "rispetto alla spread dei punti trackati. "
                            "1.5 = crop 1.5x più grande della distanza tra i punti. "
                            "Aumenta se il volto viene tagliato."
                        ),
                    },
                ),
                "face_median_window": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 21,
                        "step": 2,
                        "tooltip": (
                            "Finestra temporal median sulla face crop (frame). "
                            "1 = disabilitato. "
                            "5 = mediana su 5 frame (rimuove foglie/ombre transitorie). "
                            "Valori più alti rimuovono occlusioni più lunghe "
                            "ma possono sfumare movimenti reali veloci."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES  = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX,")
    RETURN_NAMES  = (
        "pose_data",
        "face_images",
        "key_frame_body_points",
        "bboxes",
        "face_bboxes",
    )
    FUNCTION  = "process"
    CATEGORY  = "WanAnimatePreprocess"
    DESCRIPTION = (
        "PoseAndFaceDetection con tracking manuale dei punti. "
        "Shift+click per aggiungere punti sul soggetto (volto, corpo, mani). "
        "YOLO+ViTPose vengono vincolati alla ROI definita dai punti trackati."
    )

    def process(
        self,
        model,
        images,
        width,
        height,
        tracking_points,
        roi_padding,
        smooth_keypoints,
        face_crop_scale,
        face_median_window,
    ):
        n_frames = int(images.shape[0])
        pbar     = comfy.utils.ProgressBar(n_frames)

        # Salva sempre il primo frame per il canvas
        first_frame_info = save_first_frame(images)

        # Parsa i punti
        try:
            points = json.loads(tracking_points) if tracking_points.strip() else []
        except json.JSONDecodeError:
            points = []
            print("[TrackedPose] WARNING: tracking_points non e' JSON valido, "
                  "ignorato.")

        base_node = _PoseNode()
        base_fn   = getattr(base_node, base_node.FUNCTION)

        # ── Nessun punto: detection normale ──────────────────────────────────
        if len(points) < 2:
            if points:
                print("[TrackedPose] Meno di 2 punti selezionati — "
                      "detection normale (aggiungi altri punti).")
            else:
                print("[TrackedPose] Nessun punto — detection normale.")

            raw = base_fn(model, images, width, height)
            pose_data, face_imgs, kfbp, bboxes, face_bboxes = \
                _unpack_base_result(raw)
            pose_data = _smooth_pose_data(pose_data, smooth_keypoints)
            # Senza tracking: applica solo temporal median sui bbox del nodo base
            if face_median_window > 1:
                face_imgs = _apply_pixel_median(face_imgs, face_median_window)
            pbar.update(n_frames)

            return {
                "ui":     {"first_frame": [first_frame_info]},
                "result": (pose_data, face_imgs, kfbp, bboxes, face_bboxes),
            }

        # ── Tracking + ROI mask ───────────────────────────────────────────────
        print(f"[TrackedPose] Tracking {len(points)} punti su {n_frames} frame…")
        frames_np = (images.cpu().numpy() * 255).astype(np.uint8)
        tracked   = track_optical_flow(frames_np, points)    # [N, P, 2]

        masked_np = apply_roi_masks(frames_np, tracked, roi_padding)
        masked_t  = torch.from_numpy(masked_np.astype(np.float32) / 255.0)

        # Detection sul video mascherato
        raw = base_fn(model, masked_t, width, height)
        pose_data, _face_ref, kfbp, bboxes, face_bboxes_det = \
            _unpack_base_result(raw)
        pose_data = _smooth_pose_data(pose_data, smooth_keypoints)

        # Face crop: bbox fisso dai tracked points + temporal median (opzione 4)
        # Restituisce anche i bbox fissi da usare come face_bboxes output
        # (sostituisce i bbox variabili del nodo base → mask stabile)
        face_imgs, tracked_face_bboxes = extract_face_crops_tracked_median(
            images,
            tracked,
            face_median_window=face_median_window,
            face_size=512,
            crop_scale=face_crop_scale,
        )

        pbar.update(n_frames)

        return {
            "ui":     {"first_frame": [first_frame_info]},
            "result": (pose_data, face_imgs, kfbp, bboxes, tracked_face_bboxes),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Registration
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "TrackedPoseAndFaceDetection": TrackedPoseAndFaceDetection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrackedPoseAndFaceDetection": "Tracked Pose and Face Detection",
}
