import json
import os
import re
import torch
import numpy as np
from typing import Union, List, Set, Tuple, Dict
from abc import ABC, abstractmethod
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

class ObjectDetector(ABC):
    """Abstract base class for object detection models"""
    
    @abstractmethod
    def detect(self, image: Image.Image, target_objects: Union[str, List[str]], prompt_prefix: str = "") -> Tuple[List[float], List[List[int]], List[str]]:
        """Detect objects in an image"""
        pass

    @abstractmethod
    def load_model(self):
        """Load the model into memory"""
        pass

class YOLODetector(ObjectDetector):
    """YOLO implementation of object detector"""
    
    def __init__(self, model_name: str):
        """
        Initialize YOLO detector
        Args:
            model_name: Full model name (e.g., 'yolov8n', 'yolo11x')
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading YOLO model: {self.model_name}")
            # For YOLOv8, use the direct model name
            if self.model_name.startswith('yolov8'):
                model_path = self.model_name
            # For YOLO v11, use just the number without 'yolo' prefix
            elif self.model_name.startswith('yolo11'):
                model_path = f"yolo11{self.model_name[-1]}"  # Extract size (n,s,m,l,x)
            else:
                model_path = f"ultralytics/{self.model_name}"
                
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model {self.model_name}. Error: {str(e)}")

    def detect(self, image: Image.Image, target_objects: Union[str, List[str]]) -> Tuple[List[float], List[List[int]], List[str]]:
        """
        Detect objects using YOLO
        Args:
            image: Input image
            target_objects: String or list of strings of object classes to detect
        """
        if isinstance(target_objects, str):
            target_objects = [target_objects]
        
        target_objects = [obj.lower() for obj in target_objects]
            
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        results = self.model(image_np)
        
        bboxes = []
        labels = []
        rewards = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = r.names[int(box.cls[0])]
                
                if "*" in target_objects or cls.lower() in target_objects:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    conf = float(box.conf[0])
                    
                    bboxes.append(bbox)
                    labels.append(cls)
                    rewards.append(1 - conf)  # Convert confidence to reward (lower is better)

        return rewards, bboxes, labels

class FlorenceDetector(ObjectDetector):
    """Florence model implementation of object detector"""
    
    def __init__(self, model_name: str):
        """
        Initialize Florence detector
        Args:
            model_name: Model name ('Florence-base' or 'Florence-large')
        """
        self.model_size = 'base' if 'base' in model_name.lower() else 'large'
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.load_model()

    def load_model(self):
        """Load Florence model and processor"""
        model_dir = f"/hf_cache/microsoft/Florence-2-{self.model_size}"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            local_files_only=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.model.eval()

    def detect(self, image: Image.Image, target_objects: Union[str, List[str]]) -> Tuple[List[float], List[List[int]], List[str]]:
        """Detect objects using Florence"""
        # Generate text prompt from target objects provided. The special case
        # of * should allow Florence-2 to detect any object which seems to
        # require just using its plain object detection functionality.
        if target_objects == "*" or (isinstance(target_objects, list) and "*" in target_objects):
            text = "<OD>"
            task = "<OD>"
        elif isinstance(target_objects, list):
            # We would add prompt_prefix as an argument to this function
            joined = " or ".join(target_objects)
            if prompt_prefix:
                text = f"<CAPTION_TO_PHRASE_GROUNDING> {prompt_prefix} of a {joined}"
            else:
                text = f"<CAPTION_TO_PHRASE_GROUNDING> {joined}"
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
        else:
            text = f"<CAPTION_TO_PHRASE_GROUNDING> {target_objects}"
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            
        # Resize image for Florence
        new_width = image.width // 8
        new_height = image.height // 8
        width_correction = image.width / new_width
        height_correction = image.height / new_height
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        inputs = self.processor(
            text=text,
            images=resized_image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                num_beams=3
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(resized_image.width, resized_image.height)
        )

        print(f"Raw results: ")
        print(f"{results}")
        bboxes = results[task]['bboxes']
        labels = results[task]['labels']

        # Correct bounding box coordinates
        for bbox in bboxes:
            bbox[0] = int(bbox[0] * width_correction)
            bbox[1] = int(bbox[1] * height_correction)
            bbox[2] = int(bbox[2] * width_correction)
            bbox[3] = int(bbox[3] * height_correction)

        rewards = []
        for bbox in bboxes:
            area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (image.width * image.height)
            rewards.append(area_ratio)

        return rewards, bboxes, labels

    def caption(self, image: Image.Image) -> str:
        """
        Generates a detailed caption for a given image.
        """
        task_prompt = '<MORE_DETAILED_CAPTION>'
        
        # We use more tokens here to allow for longer, more descriptive captions.
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # The model output includes the prompt, so we split it off to get only the caption.
        caption = generated_text.split(task_prompt)[-1].strip()
        return caption

class DetectorFactory:
    """Factory class to create appropriate object detector"""
    
    _class_mappings = None
    
    @staticmethod 
    def _load_class_mappings() -> dict:
        """Load class mappings from JSON file"""
        if DetectorFactory._class_mappings is None:
            json_path = os.path.join(os.path.dirname(__file__), 'yolo_classes.json')
            with open(json_path, 'r') as f:
                DetectorFactory._class_mappings = json.load(f)
        return DetectorFactory._class_mappings

    @staticmethod
    def get_model_classes(model_name: str) -> set:
        """Get set of classes supported by given model"""
        mappings = DetectorFactory._load_class_mappings()
        
        if '-oiv7' in model_name.lower():
            return set(mappings['yolo_oiv7'])
        elif 'yolo' in model_name.lower():
            return set(mappings['standard_yolo'])
        else:  # Florence can detect anything
            return set()  # Empty set means it can detect anything

    @staticmethod
    def validate_objects_for_model(model_name: str, target_objects: Union[str, List[str]]) -> bool:
        """
        Check if specified objects can be detected by the model
        Returns True if at least one object can be detected
        """
        if isinstance(target_objects, str):
            target_objects = [obj.strip() for obj in target_objects.split(',')]
            
        target_objects = [obj.lower() for obj in target_objects]
        
        # If "*" is specified, any model can detect it
        if "*" in target_objects:
            return True
            
        # Get valid classes for the model
        valid_classes = {cls.lower() for cls in DetectorFactory.get_model_classes(model_name)}
        
        # Florence can detect any object
        if 'florence' in model_name.lower():
            return True
            
        # Check if any of the target objects are in the valid classes
        return any(obj in valid_classes for obj in target_objects)

    @staticmethod
    def create_detector(model_name: str, target_objects: Union[str, List[str]]) -> 'ObjectDetector':
        """
        Create and return appropriate detector based on model name and validation
        Args:
            model_name: Full model name (e.g., 'yolov8n', 'yolo11n', 'Florence-base', 'yolov8n-oiv7')
            target_objects: Objects to detect
        """
        model_name = model_name.lower()
        
        # First validate if the model can detect the objects
        if not DetectorFactory.validate_objects_for_model(model_name, target_objects):
            print(f"Warning: {model_name} cannot detect any of the specified objects.")
            print("Falling back to Florence-base model which has broader detection capabilities.")
            return FlorenceDetector("Florence-base")
            
        # If validation passes, create the requested detector
        yolo_pattern = re.compile(r'^(?:yolov(?:8|9|10)|yolo11)[nsmlex]$')
        yolo_oiv7_pattern = re.compile(r'^yolov8[nsmlex]-oiv7$')
        florence_pattern = re.compile(r'^florence-(base|large)$', re.IGNORECASE)
        
        if 'yolo' in model_name:
            if '-oiv7' in model_name:
                if not yolo_oiv7_pattern.match(model_name):
                    raise ValueError("Invalid YOLO OIV7 model name. Must be: yolov8[n,s,m,l,x]-oiv7")
            elif not yolo_pattern.match(model_name):
                raise ValueError(
                    "Invalid YOLO model name. Must be:\n"
                    "- yolov8[n,s,m,l,x] for YOLOv8\n"
                    "- yolov9[n,s,m,l,x] for YOLOv9\n"
                    "- yolov10[n,s,m,l,x] for YOLOv10\n"
                    "- yolo11[n,s,m,l,x] for YOLOv11"
                )
            return YOLODetector(model_name)
            
        elif 'florence' in model_name:
            if not florence_pattern.match(model_name):
                raise ValueError("Invalid Florence model name. Must be: Florence-base or Florence-large")
            return FlorenceDetector(model_name)
            
        else:
            raise ValueError(
                "Invalid model type. Must be either:\n"
                "- YOLO (e.g., 'yolov8n', 'yolo11n')\n"
                "- YOLO OIV7 (e.g., 'yolov8n-oiv7')\n"
                "- Florence (e.g., 'Florence-base')"
            )

def get_label_from_image_and_object(
    image: Image.Image,
    target_object: str,
    detector: ObjectDetector,
    prompt_prefix: str = "" 
) -> List[Dict]:
    """
    Unified interface for object detection
    Returns: List of dictionaries with 'reward', 'bbox', and 'label' keys
    """
    rewards, bboxes, labels = detector.detect(image, target_object)
    
    # Convert to list of dictionaries
    results = []
    for reward, bbox, label in zip(rewards, bboxes, labels):
        results.append({
            'reward': reward,
            'bbox': bbox,
            'label': label
        })
    
    if not results:
        return []
        
    return results
