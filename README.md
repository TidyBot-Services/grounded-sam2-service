# Grounded SAM 2 Service

Open-vocabulary object detection + segmentation backend for TidyBot. Combines **Grounding DINO** (text-conditioned detection) with **SAM 2** (high-quality segmentation) on GPU (RTX 5090).

## Service URL

```
http://158.130.109.188:8001
```

## Quick Start (Client)

No external dependencies — the client uses only Python stdlib (`urllib`).

```python
from service_clients.grounded_sam2.client import GroundedSAM2Client

client = GroundedSAM2Client()  # default: http://158.130.109.188:8001

# Detect objects by text description
detections = client.detect("photo.jpg", prompts=["red cup", "screwdriver"])
for d in detections:
    print(f"{d['label']} ({d['confidence']:.2f})")

# Detect with segmentation masks
result = client.detect_full("photo.jpg", prompts=["red cup"], return_masks=True)
for d in result["detections"]:
    print(f"{d['label']}: {len(d['mask'])} polygon(s)")

# Health check
print(client.health())
```

## API Reference

### `GET /health`

Returns service status and GPU info.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "gpu_memory_mb": 32084,
  "models_loaded": true
}
```

### `POST /detect`

Run open-vocabulary detection + optional segmentation.

**Request body:**
```json
{
  "image": "<base64-encoded image>",
  "prompts": ["red cup", "screwdriver"],
  "conf": 0.3,
  "return_masks": true,
  "mask_format": "polygon"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | required | Base64-encoded JPEG or PNG |
| `prompts` | list[str] | required | Text descriptions of objects to detect |
| `conf` | float | 0.3 | Confidence threshold (0-1) |
| `return_masks` | bool | true | Run SAM 2 segmentation |
| `mask_format` | string | "polygon" | `polygon`, `rle`, or `bitmap` |

**Response:**
```json
{
  "detections": [
    {
      "bbox": {"x1": 100.5, "y1": 200.3, "x2": 300.0, "y2": 400.0},
      "confidence": 0.87,
      "label": "red cup",
      "mask": [[[100, 200], [105, 201], ...]]
    }
  ],
  "device": "cuda",
  "inference_ms": 245.3,
  "image_width": 1920,
  "image_height": 1080,
  "has_masks": true
}
```

### Mask Formats

- **polygon** — List of polygons, each a list of `[x, y]` coordinate pairs
- **rle** — Run-length encoding: `{"counts": [n0, n1, ...], "size": [H, W]}`
- **bitmap** — Base64-encoded PNG of the binary mask

## Models

- **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) — text-conditioned object detection
- **SAM 2.1** (`sam2.1_hiera_large`) — high-quality segmentation

## Running the Service

```bash
cd /home/qifei/grounded-sam2-service
pip install -r requirements.txt
# Install SAM 2 and download checkpoints (see setup below)
python main.py
```

### Setup

```bash
# Install SAM 2 from GitHub
pip install git+https://github.com/facebookresearch/sam2.git

# Install Grounding DINO via transformers
pip install transformers

# Download SAM 2.1 checkpoint
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Architecture

```
Client → POST /detect (image + text prompts)
       → Grounding DINO (text-conditioned bbox detection)
       → SAM 2 (bbox-prompted mask segmentation)
       → Response (bboxes + masks + confidence + labels)
```
