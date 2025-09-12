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
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects 'animal,bird,deer' --username <user> --password '<pass>' --cameraip <ip>
```

### Manual Context Prompt

You can provide your own context to the model using the `--prompt_prefix` argument to guide detections:

```bash
sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest --model Florence-base --objects 'animal,bird' --prompt_prefix 'A photo from a trail camera in a wilderness environment' --username <user> --password '<pass>' --cameraip <ip>
```


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

## Environment Variables
1. `PLANTNET_API_KEY` — for PlantNet API calls
2. `BLUR_MIN` — Laplacian variance threshold to trigger a focus retry (default ~120)
3. `SPECIES_MIN_SCORE` — minimum PlantNet confidence to treat as “confident” (e.g., 0.25)

## Results & Observations

When you launch the app (either with python main.py … or via the Docker command shown above), you’ll see three kinds of outputs:

1. Console logs
2. Saved images under /imgs (mount this to a host folder to persist)
3. Published messages (via Waggle plugin) containing detections & species results

### 1. Console Logs
You should see a sequence like:
- Scene caption (Florence only, if enabled)
  ```bash
  Generating dynamic context caption for the scene...
  Scene Context: "The image shows a red fence with multiple rows of small holes..."
  ```
- PTZ sweep & detection
  ```bash
    Trying PTZ: 0 0 1
    Published detection: ptz.detection.p0t0z1
    Plant detected (trees). Starting species identification workflow...
  ```
- Centering & zoom math (in degrees) and a best-of-N capture with blur score
  ```bash
  CAMERA MOVEMNET
  zoom_level: 
  current_h_fov: 
  current_v_fov:
  Move the camera to center the object
  Pan:
  Tilt:
  
  Taking final snapshot(s) for PlantNet... (Example output)
  [PLANTNET] using image -> /imgs/50.19,-13.11,11.38_plantnet_try_2025-09-10_23:19:17.327619.jpg (blur=9242.4)
  ```
- PlantNet result (success)
  ```bash
  Species: Quercus garryana
  Common Names: ['Garry oak', 'Oregon oak', 'Oregon white oak']
  Score: 0.3217
  ```
- PlantNet error example (no match / 404) (Does not publish misclassification)
  ```bash
  PlantNet identification failed: PlantNet API request failed with status 404: {"statusCode":404,"error":"Not Found","message":"Species not found"}
  ```

### 2) Saved Images
All captured frames land in /imgs inside the container. Mount it to your host to persist.
Filename format: 
```bash
  <pan>,<tilt>,<zoom>_<action>_<YYYY-MM-DD_HH:MM:SS.ffffff>.jpg
```
Without --keepimages, interim candidates may be cleaned up; the final selection is kept when you mount /imgs

### 3) Published Messages (Waggle) - Sample output
- Scene caption (Florence): ptz.scene.caption — free-text description
- Per-position detection: ptz.detection.p{pan}t{tilt}z{zoom}
- Blur/sharpness telemetry: ptz.image.blur
- PlantNet species (if any): ptz.plantnet.species
- Plain score: ptz.plantnet.score
- Alerts (optional, via alert_system.py): ptz.alert.<ALERT_TYPE> with the species JSON

### What a “Good” Run Looks Like

- Multiple `Trying PTZ: …` lines per iteration
- At least one `ptz.detection.p...` with confidence ≥ your `--confidence`
- For plant labels: centering/zoom logs, blur telemetry, PlantNet success block or a clear error
- Images written to `imgs/` (mount or `--keepimages`)

### Troubleshooting
- No species shown: PlantNet may return 404/no match. Ensure `PLANTNET_API_KEY` is set; improve view (more leaves/flowers, less backlight), increase `--species_zoom`, or adjust framing.
- Detections but no centering/zoom: Detection didn’t pass `--confidence`. Lower it slightly or ensure your `--objects` include plant terms (plant,tree,flower,bush,wildflower…).
- No images on host: Mount `/imgs` (`-v "$(pwd)/imgs":/imgs`) or use `--keepimages`.
- Soft images (low blur): Increase settle delays, try a larger `--species_zoom`, or lower `BLUR_MIN` to reduce retries.

### How It Works
- Detect objects with YOLO or Florence-2.
- Route plants via a keyword map (tree, bush, flower, plant, …).
- Center & maximize the bbox using FOV-based pan/tilt and relative zoom.
- Best-of-N capture with Laplacian variance; pick the sharpest (optionally focus-jiggle retry).
- PlantNet identify and publish results + blur telemetry + optional alerts.
