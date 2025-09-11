import os
import time
import json
import requests
from typing import Dict, Optional

PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "")
BASE_URL = os.getenv("PLANTNET_BASE_URL", "https://my-api.plantnet.org")
IDENTIFY_PATH = "/v2/identify/all"  # canonical, tolerant endpoint

# Simple retry settings for transient server issues
_RETRY_COUNT = 2
_RETRY_BACKOFF_SEC = 2.0
_TIMEOUT_SEC = 45


def _post_with_retry(url: str, params: dict, files, data: Optional[dict] = None) -> requests.Response:
    last_exc = None
    for attempt in range(_RETRY_COUNT + 1):
        try:
            return requests.post(url, params=params, files=files, data=data, timeout=_TIMEOUT_SEC)
        except requests.RequestException as e:
            last_exc = e
            if attempt < _RETRY_COUNT:
                time.sleep(_RETRY_BACKOFF_SEC * (attempt + 1))
            else:
                raise
    # Shouldn't reach here
    raise last_exc  # type: ignore


def identify_plant(image_path: str) -> Dict:
    """
    Identify a plant from a single image using PlantNet.

    Returns a dict like:
      {
        "species": str | None,
        "common_names": list[str],
        "score": float,
        "raw": <full API JSON or {"message": "no_match"}>
      }
    On failure or no match, returns {} (so callers can treat as "no result").
    """
    if not PLANTNET_API_KEY:
        # Keep this a hard error so misconfiguration is obvious
        raise RuntimeError("PLANTNET_API_KEY is not set")

    url = f"{BASE_URL}{IDENTIFY_PATH}"
    params = {"api-key": PLANTNET_API_KEY}

    # IMPORTANT: do NOT send 'organs' for now (you requested to skip it)
    # The API accepts multiple images under the same 'images' field; we send just one.
    try:
        with open(image_path, "rb") as f:
            files = [("images", (os.path.basename(image_path), f, "image/jpeg"))]
            resp = _post_with_retry(url, params=params, files=files)
    except FileNotFoundError:
        return {}

    # Graceful handling of common non-match response
    if resp.status_code == 404 and "Species not found" in resp.text:
        return {"species": None, "common_names": [], "score": 0.0, "raw": {"message": "no_match"}}

    # For all other statuses, bail out quietly (let the pipeline continue)
    if not resp.ok:
        # If you prefer raising here instead of swallowing, change to: resp.raise_for_status()
        return {}

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return {}

    results = data.get("results") or []
    if not results:
        return {"species": None, "common_names": [], "score": 0.0, "raw": {"message": "no_match"}}

    top = results[0]
    score = float(top.get("score", 0.0))
    species_name = None
    common = []

    try:
        species_name = top["species"].get("scientificNameWithoutAuthor") or top["species"].get("scientificName")
    except Exception:
        species_name = None

    try:
        cn = top["species"].get("commonNames") or []
        if isinstance(cn, list):
            common = cn
    except Exception:
        common = []

    return {
        "species": species_name,
        "common_names": common,
        "score": round(score, 4),
        "raw": data,
    }
