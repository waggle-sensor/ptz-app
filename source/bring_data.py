import os
import shutil
import logging
import math
import random
import time
import datetime
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from source import sunapi_control as camera_control
from source.object_detector import get_label_from_image_and_object

from waggle.plugin import Plugin

logger = logging.getLogger(__name__)


def draw_detections_on_image(image, detections, confidence_threshold):
    """
    Draw bounding boxes and labels on image for detections above threshold
    Args:
        image: PIL Image
        detections: List of detection dictionaries with 'bbox', 'label', 'reward' keys
        confidence_threshold: Minimum confidence to draw (reward = 1 - confidence)
    Returns:
        PIL Image with drawn boxes
    """
    # Create a copy to avoid modifying the original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for detection in detections:
        reward = detection['reward']
        confidence = 1 - reward
        
        # Only draw detections above threshold
        if confidence >= confidence_threshold:
            bbox = detection['bbox']
            label = detection['label']
            
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label background
            label_text = f"{label}: {confidence:.2f}"
            
            # Get text bounding box
            try:
                bbox_text = draw.textbbox((x1, y1), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(label_text, font=font)
            
            # Draw background rectangle for text
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill="red"
            )
            
            # Draw text
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill="white", font=font)
    
    return img_copy


try:
    # ! Note this assumes the code is running in a container
    tmp_dir = Path("/imgs")
    tmp_dir.mkdir(exist_ok=True, mode=0o777)
except OSError:
    logger.warning(
        "Could not create directories, will use default paths and the code might break"
    )

# Dictionary to store metadata for images (keyed by filename)
image_metadata = {}

def center_and_maximize_object(args, bbox, image, reward=None, label=None, increment_id=None, model_name=None):
    x1, y1, x2, y2 = bbox
    image_width, image_height = image.size
    
    print(f'x1: {x1}')
    print(f'y1: {y1}')
    print(f'x2: {x2}')
    print(f'y2: {y2}')
    print(f'image_width: {image_width}')
    print(f'image_height: {image_height}')

    # Calculate the center of the bounding box
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    
    # Calculate the center of the image
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    # Calculate the difference between the centers in pixels
    diff_x = image_center_x - bbox_center_x
    diff_y = image_center_y - bbox_center_y

    if diff_x < 0:
        print('MOVE RIGHT')
    else:
        print('MOVE LEFT')
    
    if diff_y < 0:
        print('MOVE DOWN')
    else:
        print('MOVE UP')
    
    try:
        Camera1 = camera_control.CameraControl(
            args.cameraip, args.username, args.password
        )
    except Exception as e:
        logger.error("Error when getting camera: %s", e)

    _, _, zoom_level = Camera1.requesting_cameras_position_information()
    print(f'zoom_level: {zoom_level}')

    # Get current FOV based on zoom level
    current_h_fov, current_v_fov = get_fov_from_zoom(zoom_level)
    print('current_h_fov: ', current_h_fov)
    print('current_v_fov: ', current_v_fov)
    
    # Convert pixel difference to degrees
    pan = -(diff_x / image_width) * current_h_fov
    tilt = -(diff_y / image_height) * current_v_fov

    # Move the camera to center the object
    print('Move the camera to center the object')
    print(f'Pan: {pan}')
    print(f'Tilt: {tilt}')
    try:
        Camera1.relative_control(pan=pan, tilt=tilt)
    except Exception as e:
        logger.error("Error when setting relative position: %s", e)
    
    # Calculate the current size of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Calculate the zoom factor to maximize the object size
    zoom_factor_x = image_width / bbox_width
    zoom_factor_y = image_height / bbox_height
    zoom_factor = min(zoom_factor_x, zoom_factor_y)
    print('zoom_factor_x: ', zoom_factor_x)
    print('zoom_factor_y: ', zoom_factor_y)
    print('zoom_factor: ', zoom_factor)
    
    # Calculate relative zoom
    mz=1
    MZ=40
    current_zoom_factor = zoom_level / MZ
    target_zoom_factor = current_zoom_factor * zoom_factor
    relative_zoom = target_zoom_factor * (MZ - mz) - zoom_level
    print('current_zoom_factor: ', current_zoom_factor)
    print('target_zoom_factor: ', target_zoom_factor)
    
    # Apply zoom (ensuring we don't exceed the maximum zoom)
    print('Apply zoom (ensuring we don\'t exceed the maximum zoom)')
    print(f'Relative zoom: {relative_zoom}')
    try:
        Camera1.relative_control(pan=0, tilt=0, zoom=relative_zoom)
    except Exception as e:
        logger.error("Error when setting relative position: %s", e)

    if reward is not None and reward < (1 - args.confidence) and label is not None:
        global image_metadata
        confidence = 1 - reward
        
        # Replace spaces in label with underscores for filename
        safe_label = label.replace(' ', '_')
        
        # Use increment_id in filename instead of PTZ string
        if increment_id:
            filename = f"{increment_id}_{safe_label}_conf{confidence:.2f}.jpg"
        else:
            filename = f"{safe_label}_conf{confidence:.2f}.jpg"
        
        image_path = os.path.join(tmp_dir, filename)
        
        try:
            # Take snapshot first
            Camera1.snap_shot(image_path)
            
            # Then rename with timestamp from when image was taken
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            if increment_id:
                new_filename = f"{increment_id}_{safe_label}_conf{confidence:.2f}_{timestamp}.jpg"
            else:
                new_filename = f"{safe_label}_conf{confidence:.2f}_{timestamp}.jpg"
            
            new_image_path = os.path.join(tmp_dir, new_filename)
            os.rename(image_path, new_image_path)
            
            # Store metadata for this image (all values must be strings for pywaggle)
            metadata = {
                "class": safe_label,
                "score": f"{confidence:.2f}"
            }
            if model_name:
                metadata["model"] = model_name
            image_metadata[new_filename] = metadata
        except Exception as e:
            logger.error("Error saving detection image: %s", e)

def center_and_maximize_objects_absolute(
        args, 
        detections, 
        image,
        increment_id=None
    ):
    # Get camera current absolute position and zoom level
    try:
        Camera1 = camera_control.CameraControl(
            args.cameraip, args.username, args.password
        )
    except Exception as e:
        logger.error("Error when getting camera: %s", e)

    current_pan, current_tilt, current_zoom = Camera1.requesting_cameras_position_information()
    print(f'zoom_level: {current_zoom}')

    # Get current FOV based on zoom level
    current_h_fov, current_v_fov = get_fov_from_zoom(current_zoom)
    print('current_h_fov: ', current_h_fov)
    print('current_v_fov: ', current_v_fov)

    absolute_positions = []
    zoom_levels = []
    labels_confidences_models = []
    
    for detection in detections:
        # compute all of the absolute positions of the detections
        # and the required zoom level to maximize the object size
        bbox = detection['bbox']
        reward = detection['reward']
        label = detection['label']
        model_name = detection.get('model', None)

        if reward > (1 - args.confidence):
            continue

        
        x1, y1, x2, y2 = bbox
        image_width, image_height = image.size
        
        
        print(f'x1: {x1}')
        print(f'y1: {y1}')
        print(f'x2: {x2}')
        print(f'y2: {y2}')
        print(f'image_width: {image_width}')
        print(f'image_height: {image_height}')

        # Calculate the center of the bounding box
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Calculate the center of the image
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        
        # Calculate the difference between the centers in pixels
        diff_x = image_center_x - bbox_center_x
        diff_y = image_center_y - bbox_center_y

        if diff_x < 0:
            print('MOVE RIGHT')
        else:
            print('MOVE LEFT')
        
        if diff_y < 0:
            print('MOVE DOWN')
        else:
            print('MOVE UP')
    
    
        # Convert pixel difference to degrees
        pan = -(diff_x / image_width) * current_h_fov
        tilt = -(diff_y / image_height) * current_v_fov

        absolute_pan = current_pan + pan
        absolute_tilt = current_tilt + tilt

        if absolute_pan > 360:
            absolute_pan = absolute_pan - 360
        elif absolute_pan < 0:
            absolute_pan = absolute_pan + 360

        if absolute_tilt > 90:
            absolute_tilt = 90
        elif absolute_tilt < -20:
            absolute_tilt = -20


        absolute_positions.append((absolute_pan, absolute_tilt))

    
        # Calculate the current size of the bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
    
        # Calculate the zoom factor to maximize the object size
        zoom_factor_x = image_width / bbox_width
        zoom_factor_y = image_height / bbox_height
        zoom_factor = min(zoom_factor_x, zoom_factor_y)
        print('zoom_factor_x: ', zoom_factor_x)
        print('zoom_factor_y: ', zoom_factor_y)
        print('zoom_factor: ', zoom_factor)
        
        # Calculate relative zoom
        mz=1
        MZ=40
        current_zoom_factor = current_zoom / MZ
        target_zoom_factor = current_zoom_factor * zoom_factor
        relative_zoom = target_zoom_factor * (MZ - mz) - current_zoom
        print('current_zoom_factor: ', current_zoom_factor)
        print('target_zoom_factor: ', target_zoom_factor)
        
        absolute_zoom = current_zoom + relative_zoom
        if absolute_zoom > 40:
            absolute_zoom = 40
        elif absolute_zoom < 1:
            absolute_zoom = 1

        zoom_levels.append(absolute_zoom)
        
        confidence = 1 - reward
        labels_confidences_models.append((label, confidence, model_name))
    
    global image_metadata
    for (absolute_pan, absolute_tilt), absolute_zoom, (label, confidence, model_name) in zip(absolute_positions, zoom_levels, labels_confidences_models):
        try:
            Camera1.absolute_control(absolute_pan, absolute_tilt, absolute_zoom)
            
            # Replace spaces in label with underscores for filename
            safe_label = label.replace(' ', '_')
            
            # Create filename with increment_id instead of PTZ string
            if increment_id:
                filename = f"{increment_id}_{safe_label}_conf{confidence:.2f}.jpg"
            else:
                filename = f"{safe_label}_conf{confidence:.2f}.jpg"
            
            image_path = os.path.join(tmp_dir, filename)
            
            # Take snapshot
            Camera1.snap_shot(image_path)
            
            # Rename with timestamp from when image was taken
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            if increment_id:
                new_filename = f"{increment_id}_{safe_label}_conf{confidence:.2f}_{timestamp}.jpg"
            else:
                new_filename = f"{safe_label}_conf{confidence:.2f}_{timestamp}.jpg"
            
            new_image_path = os.path.join(tmp_dir, new_filename)
            os.rename(image_path, new_image_path)
            
            # Store metadata for this image (all values must be strings for pywaggle)
            metadata = {
                "class": safe_label,
                "score": f"{confidence:.2f}"
            }
            if model_name:
                metadata["model"] = model_name
            image_metadata[new_filename] = metadata
            
        except Exception as e:
            logger.error("Error saving detection image: %s", e)

def get_image_from_ptz_position(args, object_, pan, tilt, zoom, detectors, processor, debug_detections=False, increment_id=None):
    try:
        Camera1 = camera_control.CameraControl(
            args.cameraip, args.username, args.password
        )
    except Exception as e:
        logger.error("Error when getting camera: %s", e)

    Camera1.absolute_control(pan, tilt, zoom)
    tmp_dir.mkdir(exist_ok=True, mode=0o777)

    aux_image_path = grab_image(camera=Camera1, args=args, action=0)
    image = Image.open(aux_image_path)
    os.remove(aux_image_path)

    detections = get_label_from_image_and_object(image, object_, detectors, processor)
    
    if not detections:
        LABEL = None
    else:
        # Find detection with lowest reward (highest confidence)
        best_detection = min(detections, key=lambda x: x['reward'])
        LABEL = {
            'bbox': best_detection['bbox'],
            'label': best_detection['label'],
            'reward': best_detection['reward'],
            'first': True
        }
        
        # If debug mode is enabled and there are detections above threshold, save debug image
        if debug_detections:
            detections_above_threshold = [d for d in detections if (1 - d['reward']) >= args.confidence]
            if detections_above_threshold:
                debug_image = draw_detections_on_image(image, detections_above_threshold, args.confidence)
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                # Use increment_id in debug filename if available
                if increment_id:
                    debug_filename = f"{increment_id}_debug_{timestamp}.jpg"
                else:
                    debug_filename = f"debug_{pan}_{tilt}_{zoom}_{timestamp}.jpg"
                debug_path = os.path.join(tmp_dir, debug_filename)
                debug_image.save(debug_path)
                print(f"Saved debug image: {debug_filename}")

    image_path = grab_image(camera=Camera1, args=args, action=random.randint(0,20))
    return image_path, LABEL

def get_image_from_ptz_position_multiboxes(
        args, 
        object_, 
        pan, 
        tilt, 
        zoom, 
        detectors, 
        processor,
        debug_detections=False,
        increment_id=None
    ):
    try:
        Camera1 = camera_control.CameraControl(
            args.cameraip, args.username, args.password
        )
    except Exception as e:
        logger.error("Error when getting camera: %s", e)

    Camera1.absolute_control(pan, tilt, zoom)
    tmp_dir.mkdir(exist_ok=True, mode=0o777)

    aux_image_path = grab_image(camera=Camera1, args=args, action=0)
    image = Image.open(aux_image_path)
    os.remove(aux_image_path)

    detections = get_label_from_image_and_object(image, object_, detectors, processor)
    
    # If debug mode is enabled and there are detections above threshold, save debug image
    if debug_detections and detections:
        detections_above_threshold = [d for d in detections if (1 - d['reward']) >= args.confidence]
        if detections_above_threshold:
            debug_image = draw_detections_on_image(image, detections_above_threshold, args.confidence)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            # Use increment_id in debug filename if available
            if increment_id:
                debug_filename = f"{increment_id}_debug_{timestamp}.jpg"
            else:
                debug_filename = f"debug_{pan}_{tilt}_{zoom}_{timestamp}.jpg"
            debug_path = os.path.join(tmp_dir, debug_filename)
            debug_image.save(debug_path)
            print(f"Saved debug image: {debug_filename}")

    image_path = grab_image(camera=Camera1, args=args, action=random.randint(0,20))
    return image_path, detections

def publish_images():
    global image_metadata
    with Plugin() as plugin:
        for image_file in os.listdir(tmp_dir):
            complete_path = os.path.join(tmp_dir, image_file)
            
            # Get metadata for this image if available
            meta = image_metadata.get(image_file, {})
            
            print('Publishing')
            print(complete_path)
            if meta:
                model_info = f", model={meta.get('model')}" if meta.get('model') else ""
                print(f"  Metadata: class={meta.get('class')}, score={meta.get('score')}{model_info}")
            
            plugin.upload_file(complete_path, meta=meta)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Clear metadata after publishing
    image_metadata = {}

def get_fov_from_zoom(zoom_level):
    # Camera specifications
    min_focal_length = 4.25  # mm
    max_focal_length = 170  # mm
    h_wide, h_tele = 65.66, 1.88  # degrees
    v_wide, v_tele = 39.40, 1.09  # degrees
    min_zoom, max_zoom = 1, 40  # optical zoom range

    # Ensure zoom_level is within the valid range
    zoom_level = max(min_zoom, min(max_zoom, zoom_level))

    # Calculate current focal length based on zoom level
    focal_length = min_focal_length * zoom_level

    # Calculate the sensor dimensions
    sensor_width = 2 * min_focal_length * math.tan(math.radians(h_wide / 2))
    sensor_height = 2 * min_focal_length * math.tan(math.radians(v_wide / 2))

    # Calculate current FOV
    current_h_fov = math.degrees(2 * math.atan(sensor_width / (2 * focal_length)))
    current_v_fov = math.degrees(2 * math.atan(sensor_height / (2 * focal_length)))

    return current_h_fov, current_v_fov


def grab_image(camera, args, action):
    position = camera.requesting_cameras_position_information()

    pos_str = ",".join([str(p) for p in position])
    action_str = str(action)
    # ct stores current time
    ct = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    img_path = str(tmp_dir / f"{pos_str}_{action_str}_{ct}.jpg")
    #print('img_path: ', img_path)
    try:
        camera.snap_shot(img_path)
    # TODO: need to check what kind of exception is raised
    except Exception as e:
        logger.error("Error when taking snap shot: %s : %s", img_path, e)
        #if args.publish_msgs:
            #with Plugin() as plugin:
                #plugin.publish(
                    #"cannot.capture.image.from.camera", str(datetime.datetime.now())
                #)
        return None
    return img_path

