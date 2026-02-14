"""
Grounded SAM 2 Service — TidyBot Backend
Open-vocabulary object detection (Grounding DINO) + segmentation (SAM 2) via HTTP API.
"""

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# ─── Globals ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GROUNDING_MODEL = None
SAM_PREDICTOR = None
SAM_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def load_models():
    """Load Grounding DINO and SAM 2 models."""
    global GROUNDING_MODEL, SAM_PREDICTOR

    print(f"Loading Grounding DINO ({GROUNDING_MODEL_ID}) on {DEVICE}...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GROUNDING_MODEL = {
        "processor": AutoProcessor.from_pretrained(GROUNDING_MODEL_ID),
        "model": AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL_ID).to(DEVICE),
    }

    print(f"Loading SAM 2 on {DEVICE}...")
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_model = build_sam2(SAM_MODEL_CFG, SAM_CHECKPOINT, device=DEVICE)
    SAM_PREDICTOR = SAM2ImagePredictor(sam2_model)

    print("All models loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    GROUNDING_MODEL.clear()


# ─── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="TidyBot Grounded SAM 2 Service",
    description="Open-vocabulary detection + segmentation backend for TidyBot.",
    version="0.1.0",
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────
class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    prompts: list[str] = Field(..., description="Text prompts for objects to detect (e.g. ['red cup', 'screwdriver'])")
    conf: float = Field(0.3, description="Confidence threshold (0-1)")
    return_masks: bool = Field(True, description="Return segmentation masks via SAM 2")
    mask_format: str = Field("polygon", description="Mask format: 'polygon', 'rle', or 'bitmap'")


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    bbox: BBox
    confidence: float
    label: str
    mask: Optional[list | dict | str] = Field(None, description="Segmentation mask")


class DetectResponse(BaseModel):
    detections: list[Detection]
    device: str
    inference_ms: float
    image_width: int
    image_height: int
    has_masks: bool = False


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str]
    gpu_memory_mb: Optional[int]
    models_loaded: bool


# ─── Mask Utilities ───────────────────────────────────────────────
def mask_to_polygon(mask_np: np.ndarray) -> list[list[list[float]]]:
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.squeeze().tolist()
            if isinstance(polygon[0], list):
                polygons.append(polygon)
    return polygons


def mask_to_rle(mask_np: np.ndarray) -> dict:
    pixels = mask_np.flatten().astype(np.uint8)
    runs = []
    current = 0
    count = 0
    for p in pixels:
        if p == current:
            count += 1
        else:
            runs.append(count)
            current = p
            count = 1
    runs.append(count)
    return {"counts": runs, "size": list(mask_np.shape)}


def mask_to_bitmap_b64(mask_np: np.ndarray) -> str:
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    _, png_data = cv2.imencode(".png", mask_uint8)
    return base64.b64encode(png_data.tobytes()).decode()


def format_mask(mask_np: np.ndarray, fmt: str):
    if fmt == "rle":
        return mask_to_rle(mask_np)
    elif fmt == "bitmap":
        return mask_to_bitmap_b64(mask_np)
    return mask_to_polygon(mask_np)


# ─── Endpoints ────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_name = None
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    return HealthResponse(
        status="ok",
        device=DEVICE,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
        models_loaded=GROUNDING_MODEL is not None and SAM_PREDICTOR is not None,
    )


@app.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    """Run Grounding DINO detection + optional SAM 2 segmentation."""
    # Decode image
    try:
        img_data = base64.b64decode(request.image)
        img_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    h, w = img_np.shape[:2]
    t0 = time.perf_counter()

    # ── Grounding DINO detection ──
    # Build text prompt: "red cup . screwdriver ."
    text_prompt = " . ".join(request.prompts) + " ."

    processor = GROUNDING_MODEL["processor"]
    model = GROUNDING_MODEL["model"]

    inputs = processor(images=img_pil, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=request.conf,
        text_threshold=request.conf,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu().numpy()  # (N, 4) xyxy
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]  # list of strings

    # ── SAM 2 segmentation ──
    masks_list = None
    has_masks = False
    if request.return_masks and len(boxes) > 0 and SAM_PREDICTOR is not None:
        SAM_PREDICTOR.set_image(img_np)
        # SAM2 expects (N,4) boxes
        input_boxes = boxes.copy()
        masks, scores_sam, _ = SAM_PREDICTOR.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        # masks shape: (N, 1, H, W) or (N, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks_list = masks
        has_masks = True

    inference_ms = (time.perf_counter() - t0) * 1000

    # Build detections
    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        det = Detection(
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=float(scores[i]),
            label=labels[i].strip(),
        )
        if has_masks and masks_list is not None:
            det.mask = format_mask(masks_list[i].astype(np.uint8), request.mask_format)
        detections.append(det)

    return DetectResponse(
        detections=detections,
        device=DEVICE,
        inference_ms=round(inference_ms, 2),
        image_width=w,
        image_height=h,
        has_masks=has_masks,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
