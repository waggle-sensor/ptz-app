import os
import sys
import argparse
import time
from PIL import Image
from waggle.plugin import Plugin
import json
import logging

from source.bring_data import (
    center_and_maximize_object,
    get_image_from_ptz_position,
    publish_images,
    grab_image,
)
from source import sunapi_control as camera_control
from source.object_detector import DetectorFactory, FlorenceDetector


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
        "-conf",
        "--confidence",
        help="Confidence threshold for detections (0-1, default=0.1)",
        type=float,
        default=0.1,
    )

    return parser


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
            
            dynamic_prompt_prefix = args.prompt_prefix
            if not dynamic_prompt_prefix and isinstance(detector, FlorenceDetector):
                print("Generating dynamic context caption for the scene...")
                try:
                    cam = camera_control.CameraControl(args.cameraip, args.username, args.password)
                    cam.absolute_control(pans[0], tilts[0], zooms[0])
                    temp_image_path = grab_image(camera=cam, args=args, action="caption_shot")
                    
                    if temp_image_path:
                        image = Image.open(temp_image_path)
                        dynamic_prompt_prefix = detector.caption(image)
                        os.remove(temp_image_path)
                        print(f"Scene Context: \"{dynamic_prompt_prefix}\"")
                        plugin.publish("ptz.scene.caption", dynamic_prompt_prefix)

                except Exception as e:
                    print(f"Could not generate dynamic caption: {e}")

            for pan, tilt, zoom in zip(pans, tilts, zooms):
                print(f"Trying PTZ: {pan} {tilt} {zoom}")
                image_path, detection = get_image_from_ptz_position(
                    args, objects, pan, tilt, zoom, detector, None, dynamic_prompt_prefix
                )

                if detection is None or detection["reward"] > (1 - args.confidence):
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                    continue

                detection_name = f"ptz.detection.p{pan}t{tilt}z{int(zoom)}"
                detection_payload = {
                    "label": detection["label"],
                    "confidence": 1 - detection["reward"],
                    "bbox": detection["bbox"]
                }
                plugin.publish(detection_name, json.dumps(detection_payload))
                print(f"Published detection: {detection_name}")
                print(f"Detection payload: {detection_payload}")

                label = detection["label"]
                bbox = detection["bbox"]
                reward = detection["reward"]
                confidence = 1 - reward

                print(f"Following {label} object (confidence: {confidence:.2f})")

                image = Image.open(image_path)
                center_and_maximize_object(args, bbox, image, reward, label)

                if os.path.exists(image_path):
                    os.remove(image_path)
            
            publish_images(args)

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
