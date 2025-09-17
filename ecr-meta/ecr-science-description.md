# Science
This application leverages edge computing hardware to perform real-time object detection and zooming using a PTZ (Pan‐Tilt‐Zoom) camera. By automatically detecting and focusing on objects such as people, vehicles, wildlife, or notable scene elements, the system provides key insights for ecological monitoring, urban surveillance, wildlife management, and situational awareness. Its scientific relevance lies in continuous, unobtrusive observation of dynamic environments—helping researchers and practitioners gather high-quality images for biodiversity studies, behavior research, security monitoring, and timely interventions in remote or hard-to-reach locations.

# AI@Edge
The application can deploy either **YOLO** (yolov8-yolo11n) or **Florence v2** models on the edge device. Florence v2 is a powerful vision-language model capable of identifying a more comprehensive range of objects and scenes compared to traditional object detection models.

The workflow is:
1. The camera rotates (pan/tilt) and zooms in pre-determined or incremental steps to scan the environment.  
2. Live frames are captured and processed by the selected AI model (YOLO or Florence v2).  
3. If an object of interest is detected with sufficient confidence, the system automatically adjusts the PTZ camera to center and maximize the object in the frame.
  i. After centering/zooming on plants, the app can optionally perform species identification using the PlantNet API. 
  ii. It captures several bracketed snapshots at different zooms, ranks them by sharpness (variance of Laplacian), and submits the sharpest image to PlantNet. 
  iii. Results (top candidate and optional top-K) are published as Waggle telemetry, with an optional alert if the species matches a monitored list (e.g., invasive/rare)
4. A picture is taken and sent to the cloud infrastructure for further processing, archiving, or real-time alerts.

By pushing this AI capability to the edge, the system operates continuously with minimal latency and reduced bandwidth usage—uploading only relevant snapshots rather than a constant video feed.

## Model Capabilities

### YOLO (yolo11n)
- Efficient object detection optimized for edge devices
- Detects common objects like people, vehicles, and animals
- Faster inference but more limited class recognition

### Florence v2
- Advanced vision-language model that can understand more complex scenes
- Can detect virtually any object when used with the wildcard (`*`) parameter
- Operates in `<OD>` task mode for general object detection
- More resource-intensive but provides greater detection flexibility
- Can optionally caption the scene to build context for detection
- Used in <OD> (object detection) mode for general objects, then branches to PlantNet when labels resemble plants 

### Image Quality & Focus
- Sharpness metric: variance of Laplacian (higher = sharper).
- Telemetry: publishes blur as ptz.image.blur with blur_var_laplacian.
- Refocus gate: if blur < BLUR_MIN (env), a short focus pulse is attempted before retrying.
- Settle delays: short sleeps after pan/tilt/zoom to allow AF/exposure to stabilize.

### Telemetry Topics (Waggle)
- ptz.detection.p{pan}t{tilt}z{zoom} — label, confidence, bbox, PTZ pose, timestamp
- ptz.scene.caption — optional Florence scene caption used as context
- ptz.image.blur — blur metric for the chosen PlantNet frame
- ptz.plantnet.candidates — (debug) top-K candidates with scores
- ptz.plantnet.species — final published species (gated by SPECIES_MIN_SCORE)
- ptz.plantnet.score — convenience score metric
- ptz.alert.{type} — alert on invasive/rare species (if configured)

# Arguments
The application supports the following command-line arguments:
- **`--iterations` / `-it`**  
  Number of PTZ camera rounds to run (Default: 5)
- **`--objects` / `-obj`**  
  Objects to detect (comma-separated). Use "*" to detect all objects. (Default: "person")
- **`--username` / `-un`**  
  Username for the PTZ camera
- **`--password` / `-pw`**  
  Password for the PTZ camera
- **`--cameraip` / `-ip`**  
  IP address of the PTZ camera
- **`--panstep` / `-ps`**  
  Pan step in degrees (Default: 15)
- **`--tilt` / `-tv`**  
  Tilt value in degrees (Default: 0)
- **`--zoom` / `-zm`**  
  Zoom value (Default: 1)
- **`--model` / `-m`**  
  Detection model to use: "yolo11n" or "Florence-base" (Default: "yolo11n")
- **`--iterdelay` / `-id`**  
  Minimum delay between iterations in seconds (Default: 60.0)
- **`--confidence` / `-conf`**  
  Minimum confidence threshold for detections (0-1) (Default: 0.1)
- **`--keepimages` / `-ki`**  
  Keep collected images in persistent folder for later use (Default: False)
- **`--debug`**  
  Enable debug level logging (Default: False)
- **`--prompt_prefix`**  
  Optional text prefix to add context for Florence prompts (empty = auto caption).
- **`--species_zoom`**  
  Extra relative zoom step used for species detail (Default: 10).

### Environment Variables
- **`PLANTNET_API_KEY`**  
  Required for PlantNet API calls.
- **`BLUR_MIN`**  
  Laplacian variance threshold to trigger a focus retry (default ~120).
- **`SPECIES_MIN_SCORE`**  
  Minimum PlantNet confidence to treat as “confident” (eg 0.25).

## Example Usage

### Using YOLO for Person Detection
```bash
python main.py -it 10 -obj "person" -un admin -pw secret -ip 192.168.1.100 -m yolo11n -conf 0.2
```

### Using Florence for Multiple Object Types
```bash
python main.py -it 5 -obj "person,car,dog" -un admin -pw secret -ip 192.168.1.100 -m Florence-base -conf 0.1
```

### Using Florence for General Object Detection
```bash
python main.py -it 5 -obj "*" -un username -pw 'password' -ip 130.202.23.92 -m Florence-base -conf 0.15
```

### Using Florence for Plant species detection
```bash
PLANTNET_API_KEY=... BLUR_MIN=70 SPECIES_MIN_SCORE=0.25 \
python main.py \
  -it 3 -obj "plant,tree" -un camera -pw 'secret' -ip 192.168.1.100 \
  -m Florence-base --species_zoom 10 --iterdelay 0 --debug
```

# Ontology
The interesting images collected by the system are tagged with metadata for easy retrieval and analysis. This includes:

- Object type (person, car, animal, etc.)
- Confidence score
- Timestamp
- Camera position (pan, tilt, zoom)
- Location data (if available)

In addition to existing fields, images and messages may include:
- Species (scientific)
- Common names
- PlantNet score
- Blur sharpness (blur_var_laplacian)
- Candidates (top-K species + scores, debug)
- These fields enable downstream filtering by species, quality scoring, and confidence-based triage.

This metadata enables systematic analysis of object presence, movement patterns, and temporal dynamics in the monitored environment.
