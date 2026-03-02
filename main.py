import os
import sys
import argparse
import time
from PIL import Image
from source.bring_data import (
    center_and_maximize_object,
    center_and_maximize_objects_absolute,
    get_image_from_ptz_position,
    get_image_from_ptz_position_multiboxes,
    publish_images,
)
from source.object_detector import DetectorFactory
import logging


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
        help="Model(s) to use (comma-separated for multiple, e.g., 'yolo11n' or 'yolo11n,Florence-base' or 'BioCLIP')",
        type=str,
        default="yolo11n",
    )
    parser.add_argument(
        "--bioclip-rank",
        help="Taxonomic rank for BioCLIP classification (default=Class). Options: Kingdom, Phylum, Class, Order, Family, Genus, Species",
        type=str,
        default="Class",
    )
    parser.add_argument(
        "--bioclip-taxon",
        help="Target taxon for BioCLIP to detect (default='Animalia Chordata Mammalia' for mammals). Examples: 'Animalia Chordata Aves' (birds), 'Animalia Arthropoda Insecta' (insects)",
        type=str,
        default="Animalia Chordata Mammalia",
    )
    parser.add_argument(
        "--bioclip-confidence",
        help="Confidence threshold for BioCLIP detections (0-1, default=0.3)",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "-id",
        "--iterdelay",
        help="Delay in seconds between iterations (default=0.0)",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "-conf",
        "--confidence",
        help="Confidence threshold for detections (0-1, default=0.1)",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-multiple",
        "--multiple",
        help="Save multiple images for each discrete PTZ position (default=False)",
        action="store_true",
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Publish images with detection boxes drawn for debugging",
    )

    return parser


def look_for_object(args):
    objects = [obj.strip().lower() for obj in args.objects.split(",")]
    pans = [angle for angle in range(0, 360, args.panstep)]
    tilts = [args.tilt for _ in range(len(pans))]
    zooms = [args.zoom for _ in range(len(pans))]

    # Parse model string - can be single or comma-separated list
    model_names = [model.strip() for model in args.model.split(",")]
    
    # Create detectors for each model
    detectors = []
    for model_name in model_names:
        try:
            # Pass BioCLIP-specific parameters if it's a BioCLIP model
            if 'bioclip' in model_name.lower():
                detector = DetectorFactory.create_detector(
                    model_name, 
                    args.objects,
                    bioclip_rank=args.bioclip_rank,
                    bioclip_taxon=args.bioclip_taxon,
                    bioclip_confidence=args.bioclip_confidence
                )
            else:
                detector = DetectorFactory.create_detector(model_name, args.objects)
            
            detectors.append(detector)
            print(f"Successfully loaded model: {model_name}")
        except ValueError as e:
            print(f"Error creating detector for {model_name}: {str(e)}")
            sys.exit(1)
    
    if not detectors:
        print("No valid detectors created")
        sys.exit(1)

    for iteration in range(args.iterations):
        iteration_start_time = time.time()

        for idx, (pan, tilt, zoom) in enumerate(zip(pans, tilts, zooms)):
            # Create increment ID based on initial PTZ position
            increment_id = f"scan_{pan:03d}_{tilt:03d}_{zoom:02d}"
            print(f"Trying PTZ: {pan} {tilt} {zoom} (ID: {increment_id})")

            if not args.multiple:
                image_path, detection = get_image_from_ptz_position(
                    args, objects, pan, tilt, zoom, detectors, None, args.debug_detections, increment_id
                )
                if detection is None or detection["reward"] > (1 - args.confidence):
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                    continue

                label = detection["label"]
                bbox = detection["bbox"]
                reward = detection["reward"]
                model_name = detection.get("model", None)
                confidence = 1 - reward

                model_info = f" detected by {model_name}" if model_name else ""
                print(f"Following {label} object (confidence: {confidence:.2f}){model_info}")

                image = Image.open(image_path)
                center_and_maximize_object(args, bbox, image, reward, label, increment_id, model_name)
            else:
                # get multiple images for each detection
                image_path, detections = get_image_from_ptz_position_multiboxes(
                    args, objects, pan, tilt, zoom, detectors, None, args.debug_detections, increment_id
                )
                
                if not detections:
                    if image_path and os.path.exists(image_path):
                        os.remove(image_path)
                    continue
                
                image = Image.open(image_path)
                center_and_maximize_objects_absolute(args, detections, image, increment_id)
                


            if os.path.exists(image_path):
                os.remove(image_path)

            # Publish images after each increment
            publish_images()

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
