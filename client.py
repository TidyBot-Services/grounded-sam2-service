"""
TidyBot Grounded SAM 2 Service — Python Client SDK

Usage:
    from client import GroundedSAM2Client

    client = GroundedSAM2Client("http://<backend-host>:8001")

    # Check service health
    print(client.health())

    # Detect objects by text prompts
    detections = client.detect("photo.jpg", prompts=["red cup", "screwdriver"])
    for d in detections:
        print(f"{d['label']} ({d['confidence']:.2f})")

    # Full response with masks
    result = client.detect_full("photo.jpg", prompts=["red cup"], return_masks=True, mask_format="polygon")
    for d in result["detections"]:
        print(f"{d['label']}: {len(d.get('mask', []))} polygon(s)")
"""

import base64
import requests
from pathlib import Path
from typing import Optional


class GroundedSAM2Client:
    """Client SDK for the TidyBot Grounded SAM 2 Service."""

    def __init__(self, base_url: str = "http://localhost:8001", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        """Check service health and GPU status."""
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _encode_image(self, image) -> str:
        """Encode image to base64 from file path, bytes, or pass through if already base64."""
        if isinstance(image, (str, Path)):
            return base64.b64encode(Path(image).read_bytes()).decode()
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode()
        return image

    def detect(
        self,
        image,
        prompts: list[str],
        conf: float = 0.3,
    ) -> list[dict]:
        """
        Detect objects by text prompts. Returns detections only (no masks).

        Args:
            image: File path (str/Path), raw bytes, or base64 string.
            prompts: Text descriptions of objects to find (e.g. ["red cup", "screwdriver"]).
            conf: Confidence threshold (0.0 - 1.0).

        Returns:
            List of detections with bbox, confidence, label.
        """
        payload = {
            "image": self._encode_image(image),
            "prompts": prompts,
            "conf": conf,
            "return_masks": False,
        }
        r = requests.post(f"{self.base_url}/detect", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["detections"]

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
            image: File path (str/Path), raw bytes, or base64 string.
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
        r = requests.post(f"{self.base_url}/detect", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    client = GroundedSAM2Client()
    print("Health:", client.health())
