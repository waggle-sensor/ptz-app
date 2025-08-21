# PTZ APP

This is an intelligent, autonomous PTZ camera application that uses advanced vision-language models (YOLO or Florence-2) to detect, frame, and analyze objects of interest in real-time. It is designed for deployment on edge computing nodes within the Sage project.

---

## What’s New

- **Florence-2 Enhancements**
  - Automatic scene context generation at the start of scans.
  - Support for manual prompt injection with `--prompt_prefix` to guide detections.

- **Enhanced Data Logging**
  - Publishes scene captions (Florence-2) and raw detection data (label, confidence, position) with each scan.
  - Improved debug-level logging.

- **Pipeline Updates**
  - Added Scene Analysis step (Florence-2 only).
  - Data Publishing step now explicitly includes metadata publishing.

---

## How It Works

The algorithm performs the following steps:

1. **Initialization**  
   Sets up the object detection model (`YOLO` or `Florence-2`) based on user parameters.

2. **Scene Analysis (Florence-2 Only)**  
   At the start of a scan, Florence-2 can automatically generate a detailed text caption of the current scene to use as a dynamic, contextual prompt.  

3. **Contextual Area Scanning**  
   Systematically scans the environment by rotating the PTZ camera in pan steps (default: 15°) through a full 360° rotation at the specified tilt and zoom level.  
   When using Florence-2, the generated (or user-specified) context is incorporated to improve detection relevance.

4. **Object Detection**  
   At each camera position, captures an image and runs object detection to identify specified objects (e.g., person, car, dog).

5. **Filtering**  
   Filters detections based on confidence threshold (default: 0.1).

6. **Object Tracking**  
   When an object of interest is detected with sufficient confidence, the algorithm:
   - Centers the camera on the detected object
   - Adjusts zoom to maximize the object in the frame

7. **Data Publishing**  
   Saves and publishes the optimized images of detected objects.  
   Publishes rich metadata—including scene captions (Florence-2), raw detection data (labels, confidence, position), and logging outputs—to the Sage data portal.

8. **Iteration**  
   Repeats the process for the specified number of iterations with configurable delay between scans.

---

## Build the container

```bash
sudo docker buildx build --platform=linux/amd64,linux/arm64/v8 -t your_docker_hub_user_name/ptzapp -f Dockerfile --push .
```

Then pull the container from dockerhub in the node:

```bash
sudo docker image pull your_docker_hub_user_name/ptzapp
```

## Run the container on a dell blade

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest -ki -it 5 -un camera_user_name -pw camera_password -ip camera_ip_address -obj person,car
```

## Run the container on a waggle node

```bash
sudo docker run -it --rm your_docker_hub_user_name/ptzapp:latest -ki -it 5 -un camera_user_name -pw camera_password -ip camera_ip_address -obj person,car
```

## Example with Florence model

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --iterations 5 --username username --password 'password' --cameraip 130.202.23.92 --objects 'person,car'
```
## Advanced Usage with Florence-2

### Fully Autonomous Mode (Automatic Context)

When using Florence-2 without a manual prompt, the application will automatically analyze the scene to generate its own context before searching for objects:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects 'animal,bird,deer' --username <user> --password '<pass>' --cameraip <ip>```

### Manual Context Prompt

You can provide your own context to the model using the `--prompt_prefix` argument to guide detections:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects 'animal,bird' --prompt_prefix 'A photo from a trail camera in a wilderness environment' --username <user> --password '<pass>' --cameraip <ip>```


## Using Different Object Detection Models

### YOLO (Default)
By default, the application uses the YOLO model (yolo11n) for object detection. Specify objects by name:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --objects 'person,car,dog'
```

### Florence Models
When using Florence models, you have more powerful detection capabilities:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects 'person,car'
```

### Detecting All Objects with Florence

To detect all objects using Florence models, use the asterisk:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects '*'
```

**Note:** When using `'*'` with Florence models, the application runs in the `<OD>` task mode, which enables general object detection without filtering for specific classes. This can be useful for inventorying all objects in a scene but may produce more diverse results than when targeting specific objects.

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model` | `-m` | Model to use (e.g., 'yolo11n', 'Florence-base') | yolo11n |
| `--iterations` | `-it` | Number of iterations (PTZ rounds) to run | 5 |
| `--username` | `-un` | PTZ camera username | "" |
| `--password` | `-pw` | PTZ camera password | "" |
| `--cameraip` | `-ip` | PTZ camera IP address | "" |
| `--objects` | `-obj` | Objects to detect (comma-separated or '*' for everything) | "person" |
| `--keepimages` | `-ki` | Keep collected images in persistent folder | False |
| `--panstep` | `-ps` | Step of pan in degrees | 15 |
| `--tilt` | `-tv` | Tilt value in degrees | 0 |
| `--zoom` | `-zm` | Zoom value | 1 |
| `--confidence` | `-conf` | Confidence threshold (0-1) | 0.1 |
| `--iterdelay` | `-id` | Minimum delay in seconds between iterations | 60.0 |
| `--prompt_prefix` |  | Manual text prompt for Florence-2 context | "" |
| `--debug` | | Enable debug level logging | False |
