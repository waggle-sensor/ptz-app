# Science
This application leverages edge computing hardware to perform real-time object detection and zooming using a PTZ (Pan‐Tilt‐Zoom) camera. By automatically detecting and focusing on objects such as people, vehicles, wildlife, or notable scene elements, the system provides key insights for ecological monitoring, urban surveillance, wildlife management, and situational awareness. Its scientific relevance lies in continuous, unobtrusive observation of dynamic environments—helping researchers and practitioners gather high-quality images for biodiversity studies, behavior research, security monitoring, and timely interventions in remote or hard-to-reach locations.

# AI@Edge
The application can deploy either **YOLO** (yolov8-yolo11n) or **Florence v2** models on the edge device. Florence v2 is a powerful vision-language model capable of identifying a more comprehensive range of objects and scenes compared to traditional object detection models.

The workflow is:
1. The camera rotates (pan/tilt) and zooms in pre-determined or incremental steps to scan the environment.  
2. Live frames are captured and processed by the selected AI model (YOLO or Florence v2).  
3. If an object of interest is detected with sufficient confidence, the system automatically adjusts the PTZ camera to center and maximize the object in the frame.  
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
- **`--multiple`**  
  Save multiple images for multiple detections in a single frame (Default: False)

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

### Saving multiple images per discrete sweep location
```bash
python main.py -it 5 -obj "*" -un username -pw 'password' -ip 130.202.23.92 -m Florence-base  --multiple
```

# Ontology
The interesting images collected by the system are tagged with metadata for easy retrieval and analysis. This includes:

- Object type (person, car, animal, etc.)
- Confidence score
- Timestamp
- Camera position (pan, tilt, zoom)
- Location data (if available)

This metadata enables systematic analysis of object presence, movement patterns, and temporal dynamics in the monitored environment.
