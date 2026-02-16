"""
TidyBot Grounded SAM 2 Service — Python Client SDK

Usage:
    from service_clients.grounded_sam2.client import GroundedSAM2Client

    client = GroundedSAM2Client()
    detections = client.detect(image_bytes, prompts=['red cup'])
    for d in detections:
        print(f"{d['label']} ({d['confidence']:.2f}): {d['bbox']}")
"""

import base64
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


class GroundedSAM2Client:
    """Client SDK for the TidyBot Grounded SAM 2 Service."""

    def __init__(self, host: str = "http://158.130.109.188:8001", timeout: float = 60.0):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.host}{path}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        """Check service health and GPU status."""
        return self._get("/health")

    @staticmethod
    def _encode_image(image) -> str:
        """Encode image to base64 from file path, bytes, numpy array, or pass through if already base64."""
        if isinstance(image, (str, Path)):
            p = Path(image)
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode()
            return image  # assume base64 string
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        else:
            # numpy array
            import cv2
            _, buf = cv2.imencode(".png", image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf.tobytes()).decode()

    def detect(
        self,
        image,
        prompts: list[str],
        conf: float = 0.3,
    ) -> list[dict]:
        """
        Detect objects by text prompts. Returns detections only (no masks).

        Args:
            image: File path (str/Path), raw bytes, numpy array, or base64 string.
            prompts: Text descriptions of objects to find.
            conf: Confidence threshold (0.0 - 1.0).

        Returns:
            List of dicts with keys: label, confidence, bbox {x1,y1,x2,y2}.
        """
        payload = {
            "image": self._encode_image(image),
            "prompts": prompts,
            "conf": conf,
            "return_masks": False,
        }
        return self._post("/detect", payload)["detections"]

    def detect_full(
        self,
        image,
        prompts: list[str],
        conf: float = 0.3,
        return_masks: bool = True,
        mask_format: str = "polygon",
    ) -> dict:
        """
        Detect + segment with full metadata.

        Args:
            image: File path (str/Path), raw bytes, numpy array, or base64 string.
            prompts: Text descriptions of objects to find.
            conf: Confidence threshold (0.0 - 1.0).
            return_masks: If True, run SAM 2 segmentation.
            mask_format: 'polygon', 'rle', or 'bitmap'.

        Returns:
            dict with detections, device, inference_ms, image dimensions, has_masks.
        """
        payload = {
            "image": self._encode_image(image),
            "prompts": prompts,
            "conf": conf,
            "return_masks": return_masks,
            "mask_format": mask_format,
        }
        return self._post("/detect", payload)


if __name__ == "__main__":
    client = GroundedSAM2Client()
    print("Health:", client.health())
