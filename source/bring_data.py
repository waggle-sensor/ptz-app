import os
import shutil
import logging
import math
import random
import time
import datetime
from PIL import Image
from pathlib import Path
from source import sunapi_control as camera_control
from source.object_detector import get_label_from_image_and_object

from waggle.plugin import Plugin

logger = logging.getLogger(__name__)



try:
    # ! Note this assumes the code is running in a container
    tmp_dir = Path("/imgs")
    tmp_dir.mkdir(exist_ok=True, mode=0o777)
except OSError:
    logger.warning(
        "Could not create directories, will use default paths and the code might break"
    )

def center_and_maximize_object(args, bbox, image, reward=None, label=None):
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
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        confidence = 1 - reward
        filename = f"{label}_conf{confidence:.2f}_{timestamp}.jpg"
        image_path = os.path.join(tmp_dir, filename)
        
        try:
            Camera1.snap_shot(image_path)
            return image_path
        except Exception as e:
            logger.error("Error saving detection image: %s", e)
    return None

def get_image_from_ptz_position(args, object_, pan, tilt, zoom, model, processor, prompt_prefix: str = ""):
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

    detections = get_label_from_image_and_object(image, object_, model, prompt_prefix)
    
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

    image_path = grab_image(camera=Camera1, args=args, action=random.randint(0,20))
    return image_path, LABEL

def publish_images():
    with Plugin() as plugin:
        ct = str(datetime.datetime.now())
        for image_file in os.listdir(tmp_dir):
            complete_path = os.path.join(tmp_dir, image_file)
            print('Publishing')
            print(complete_path)
            plugin.upload_file(complete_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

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

