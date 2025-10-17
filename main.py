import os
import sys
import argparse
import time
from PIL import Image
from waggle.plugin import Plugin
import json
import logging
import cv2
import numpy as np
from source import alert_system
from source.bring_data import (
    center_and_maximize_object,
    get_image_from_ptz_position,
    publish_images,
    grab_image,
)
from source import plantnet_client, sunapi_control as camera_control
from source.object_detector import DetectorFactory, FlorenceDetector

# ---- Plant label helpers
PLANT_KEYWORDS = {
    'plant','flower','tree','wildflower','bush','shrub','vegetation',
    'leaf','leaves','branch','branches','trunk','potted plant','potted'
}
LARGE_PLANT = {'tree', 'bush'}

# ---- Tunables (can override via env)
BLUR_MIN = float(os.getenv("BLUR_MIN", "70"))               # refocus threshold
SPECIES_MIN_SCORE = float(os.getenv("SPECIES_MIN_SCORE", "0.25"))  # publish gate

logger = logging.getLogger(__name__)


def get_argparser():
    parser = argparse.ArgumentParser("PTZ APP")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug level logging.",
    )
    parser.add_argument(
        "-ki",
        "--keepimages",
        action="store_true",
        help="Keep collected images in persistent folder for later use",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        help="An integer with the number of iterations (PTZ rounds) to be run (default=5).",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-obj",
        "--objects",
        help="Objects to capture with the camera (comma-separated, e.g., 'person,car,dog')",
        type=str,
        default="person",
    )
    parser.add_argument(
        "-un",
        "--username",
        help="The username of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-pw",
        "--password",
        help="The password of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-ip", "--cameraip", help="The ip of the PTZ camera.", type=str, default=""
    )
    parser.add_argument(
        "-ps", "--panstep", help="The step of pan in degrees.", type=int, default=15
    )
    parser.add_argument(
        "-tv", "--tilt", help="The tilt value in degrees.", type=int, default=0
    )
    parser.add_argument("-zm", "--zoom", help="The zoom value.", type=int, default=1)
    parser.add_argument(
        "-m",
        "--model",
        help="Model to use (e.g., 'yolo11n', 'Florence-base')",
        type=str,
        default="yolo11n",
    )
    parser.add_argument(
        "-id",
        "--iterdelay",
        help="Delay in seconds between iterations (default=0.0)",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--prompt_prefix",
        help="An optional prefix to add to the Florence prompt for context.",
        type=str,
        default="",
    )
    parser.add_argument(
        "--species_zoom",
        help="Additional relative zoom steps to apply for species identification (default=10).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-conf",
        "--confidence",
        help="Confidence threshold for detections (0-1, default=0.1)",
        type=float,
        default=0.1,
    )

    return parser


def _blur_score(path: str) -> float:
    """Variance of Laplacian: higher = sharper."""
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _plantnet_topk(raw: dict, k: int = 3):
    """Extract a compact top-k from PlantNet raw JSON."""
    out = []
    try:
        results = (raw or {}).get("results") or []
        for r in results[:k]:
            sp = r.get("species", {}) or {}
            name = sp.get("scientificNameWithoutAuthor") or sp.get("scientificName")
            out.append({
                "species": name,
                "common_names": sp.get("commonNames", []),
                "score": round(float(r.get("score", 0.0)), 4),
            })
    except Exception:
        pass
    return out


def look_for_object(args):
    objects = [obj.strip().lower() for obj in args.objects.split(",")]
    pans = [angle for angle in range(0, 360, args.panstep)]
    tilts = [args.tilt for _ in range(len(pans))]
    zooms = [args.zoom for _ in range(len(pans))]

    try:
        detector = DetectorFactory.create_detector(args.model, args.objects)
    except ValueError as e:
        print(f"Error creating detector: {str(e)}")
        sys.exit(1)

    with Plugin() as plugin:
        for iteration in range(args.iterations):
            iteration_start_time = time.time()

            # --- Dynamic scene caption for Florence (only if no manual prompt) ---
            dynamic_prompt_prefix = args.prompt_prefix
            if not dynamic_prompt_prefix and isinstance(detector, FlorenceDetector):
                print("Generating dynamic context caption for the scene...")
                try:
                    cam = camera_control.CameraControl(args.cameraip, args.username, args.password)
                    cam.absolute_control(pans[0], tilts[0], zooms[0])
                    temp_image_path = grab_image(camera=cam, args=args, action="caption_shot")
                    if temp_image_path:
                        try:
                            with Image.open(temp_image_path) as _im:
                                _im.load()
                                dynamic_prompt_prefix = detector.caption(_im)
                        finally:
                            try:
                                os.remove(temp_image_path)
                            except Exception:
                                pass
                        print(f"Scene Context: \"{dynamic_prompt_prefix}\"")
                        plugin.publish("ptz.scene.caption", dynamic_prompt_prefix)
                except Exception as e:
                    print(f"Could not generate dynamic caption: {e}")

            # --- Sweep PTZ positions ---
            for pan, tilt, zoom in zip(pans, tilts, zooms):
                print(f"Trying PTZ: {pan} {tilt} {zoom}")
                image_path, detection = get_image_from_ptz_position(
                    args, objects, pan, tilt, zoom, detector, None, dynamic_prompt_prefix
                )

                # Nothing useful? clean any temp and continue
                if detection is None or detection["reward"] > (1 - args.confidence):
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except Exception:
                            pass
                    continue

                # Publish initial detection telemetry (add raw_reward for later calibration)
                detection_name = f"ptz.detection.p{pan}t{tilt}z{int(zoom)}"
                detection_payload = {
                    "label": detection["label"],
                    "confidence": round(1 - detection["reward"], 4),
                    "raw_reward": detection["reward"],
                    "bbox": detection["bbox"],
                    "ptz_position": [pan, tilt, zoom],
                    "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                }
                plugin.publish(detection_name, json.dumps(detection_payload))
                print(f"Published detection: {detection_name}")

                # Decide path by label — load/copy so we can safely delete later
                with Image.open(image_path) as _im:
                    _im.load()
                    image = _im.copy()

                # Normalize label and handle plurals/hyphens
                label_norm = detection["label"].lower().strip()
                label_norm = label_norm.replace('-', ' ')  # "wild-flower" -> "wild flower"
                root = label_norm[:-1] if label_norm.endswith('s') else label_norm  # "trees" -> "tree"
                tokens = set(label_norm.split())
                is_plant = (
                    root in PLANT_KEYWORDS
                    or label_norm in PLANT_KEYWORDS
                    or any(tok in PLANT_KEYWORDS for tok in tokens)
                )

                if is_plant:
                    # --- PLANT WORKFLOW: center/zoom + best-of-N sharpest PlantNet shot ---
                    print(f"Plant detected ({detection['label']}). Starting species identification workflow...")
                    try:
                        # Step 1: center + frame whole plant
                        center_and_maximize_object(args, detection["bbox"], image,
                                                   detection["reward"], detection["label"])

                        # Step 2: if large plant, add an extra zoom for details
                        if root in LARGE_PLANT and args.species_zoom > 0:
                            print(f"Performing additional zoom ({args.species_zoom}) for species detail...")
                            cam = camera_control.CameraControl(args.cameraip, args.username, args.password)
                            cam.relative_control(pan=0, tilt=0, zoom=args.species_zoom)
                            time.sleep(2)

                        # Step 3: Best-of-N shots with blur gate
                        print("Taking final snapshot(s) for PlantNet...")
                        cam = camera_control.CameraControl(args.cameraip, args.username, args.password)

                        def _take_stable_shot(action_label: str, settle: float = 1.0):
                            time.sleep(settle)  # AF/exposure settle
                            return grab_image(camera=cam, args=args, action=action_label)

                        zoom_increments = [0]
                        if args.species_zoom > 0:
                            zoom_increments.extend([max(1, args.species_zoom // 2), args.species_zoom])

                        candidates = []
                        for dz in zoom_increments:
                            try:
                                if dz:
                                    cam.relative_control(pan=0, tilt=0, zoom=dz)
                                    time.sleep(0.5)
                                shot_path = _take_stable_shot("plantnet_try", settle=1.0)
                                if shot_path:
                                    candidates.append((shot_path, _blur_score(shot_path)))
                            except Exception as e:
                                print("Bracket shot failed:", e)

                        if not candidates:
                            print("No candidate images captured for PlantNet.")
                        else:
                            # pick sharpest
                            candidates.sort(key=lambda t: t[1], reverse=True)
                            final_image_path, best_blur = candidates[0]
                            print(f"[PLANTNET] using image -> {final_image_path} (blur={best_blur:.1f})")

                            # optional: one focus jiggle retry if still soft
                            if best_blur < BLUR_MIN:
                                try:
                                    cam.continuous_control(focus='Near'); time.sleep(0.3)
                                    cam.continuous_control(focus='Stop'); time.sleep(0.6)
                                    retry_path = _take_stable_shot("plantnet_refocus", settle=1.0)
                                    if retry_path:
                                        retry_blur = _blur_score(retry_path)
                                        if retry_blur > best_blur:
                                            if not args.keepimages and final_image_path and final_image_path != retry_path:
                                                try:
                                                    os.remove(final_image_path)
                                                except Exception:
                                                    pass
                                            final_image_path, best_blur = retry_path, retry_blur
                                        else:
                                            if not args.keepimages and retry_path:
                                                try:
                                                    os.remove(retry_path)
                                                except Exception:
                                                    pass
                                except Exception as e:
                                    print("Focus jiggle retry failed:", e)

                            # publish blur telemetry
                            plugin.publish("ptz.image.blur", json.dumps({
                                "blur_var_laplacian": round(best_blur, 2),
                                "ptz_position": [pan, tilt, zoom]
                            }))

                            # PlantNet identify
                            try:
                                species_results = plantnet_client.identify_plant(final_image_path)
                                if species_results:
                                    # Optional top-k telemetry from raw
                                    topk = _plantnet_topk(species_results.get("raw", {}), k=3)
                                    if topk:
                                        plugin.publish("ptz.plantnet.topk", json.dumps({
                                            "ptz_position": [pan, tilt, zoom],
                                            "candidates": topk
                                        }))

                                    # Gate publishing on score
                                    score = float(species_results.get("score", 0.0))
                                    species_name = species_results.get("species")

                                    if species_name and score >= SPECIES_MIN_SCORE:
                                        lean = {
                                            "species": species_name,
                                            "common_names": species_results.get("common_names", []),
                                            "score": round(score, 4),
                                            "blur_var_laplacian": round(best_blur, 2),
                                            "ptz_position": [pan, tilt, zoom],
                                            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                                        }
                                        print(f"Published plantnet species: {lean}")
                                        plugin.publish("ptz.plantnet.species", json.dumps(lean))
                                        plugin.publish("ptz.plantnet.score", str(lean["score"]))

                                        # Alerts on invasive/rare
                                        alert_type, _ = alert_system.check_for_alert(species_name)
                                        if alert_type:
                                            print(f"!!! ALERT: {alert_type} '{species_name}' detected! Publishing alert.")
                                            plugin.publish(f"ptz.alert.{alert_type}", json.dumps(lean))
                                    else:
                                        print(f"Species score low ({score:.2f}) or name missing; skipping publish.")
                            except Exception as e:
                                print(f"PlantNet identification failed: {e}")

                            # cleanup non-selected candidates if not keeping images
                            if not args.keepimages:
                                for p, _b in candidates[1:]:
                                    try:
                                        os.remove(p)
                                    except Exception:
                                        pass
                    except Exception as e:
                        print(f"PlantNet identification pipeline failed: {e}")
                else:
                    # --- OTHER OBJECTS: follow & maximize in frame ---
                    print(f"Following {detection['label']} object (confidence: {1 - detection['reward']:.2f})")
                    center_and_maximize_object(args, detection["bbox"], image,
                                               detection["reward"], detection["label"])

                # always clean initial temp image from this PTZ step
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass

            # publish any saved images and wait for next iteration
            publish_images(args.keepimages)

            iteration_time = time.time() - iteration_start_time
            if args.iterdelay > 0:
                remaining_delay = max(0, args.iterdelay - iteration_time)
                if remaining_delay > 0:
                    print(f"Waiting {remaining_delay:.2f} seconds before next iteration...")
                    time.sleep(remaining_delay)


def main():
    args = get_argparser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    look_for_object(args)


if __name__ == "__main__":
    main()
