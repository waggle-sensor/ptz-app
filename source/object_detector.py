import json
import os
import re
import torch
import numpy as np
import cv2
import collections
import heapq
import torch.nn.functional as F
from typing import Union, List, Set, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
from torchvision import transforms
from huggingface_hub import hf_hub_download
import open_clip


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes
    Args:
        box1, box2: [x1, y1, x2, y2] format
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def combine_detections_from_models(detections_list: List[List[Dict]], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Combine detections from multiple models, merging overlapping detections
    Args:
        detections_list: List of detection lists from different models
        iou_threshold: IoU threshold for considering boxes as duplicates
    Returns:
        Combined list of detections with duplicates merged
    """
    if not detections_list:
        return []
    
    # Flatten all detections into a single list
    all_detections = []
    for detections in detections_list:
        all_detections.extend(detections)
    
    if not all_detections:
        return []
    
    # Sort by confidence (lower reward = higher confidence)
    all_detections.sort(key=lambda x: x['reward'])
    
    # Non-maximum suppression with label matching
    final_detections = []
    
    for detection in all_detections:
        should_add = True
        bbox = detection['bbox']
        label = detection['label'].lower()
        model_name = detection.get('model', 'unknown')
        
        for existing in final_detections:
            existing_bbox = existing['bbox']
            existing_label = existing['label'].lower()
            
            # Only merge if labels are the same or similar
            if label == existing_label or label in existing_label or existing_label in label:
                iou = compute_iou(bbox, existing_bbox)
                
                if iou > iou_threshold:
                    # This detection overlaps significantly with an existing one
                    # Keep the one with better confidence (lower reward)
                    if detection['reward'] < existing['reward']:
                        # Replace existing with this better detection
                        existing['bbox'] = bbox
                        existing['reward'] = detection['reward']
                        existing['label'] = detection['label']
                        existing['model'] = model_name
                    else:
                        # Keep existing but note that multiple models detected it
                        existing_model = existing.get('model', 'unknown')
                        if existing_model != model_name and model_name not in existing_model:
                            existing['model'] = f"{existing_model},{model_name}"
                    should_add = False
                    break
        
        if should_add:
            final_detections.append(detection)
    
    return final_detections

class ObjectDetector(ABC):
    """Abstract base class for object detection models"""
    
    @abstractmethod
    def detect(self, image: Image.Image, target_objects: Union[str, List[str]]) -> Tuple[List[float], List[List[int]], List[str]]:
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
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.load_model()

    def load_model(self):
        """Load Florence model and processor"""
        model_dir = f"/hf_cache/microsoft/Florence-2-{self.model_size}"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager"
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
            joined = " or ".join(target_objects)
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
        ).to(self.device, self.dtype)

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


class BioCLIPDetector(ObjectDetector):
    """BioCLIP implementation of object detector with gradient-based localization"""
    
    # BioCLIP data files
    TXT_EMB_NPY = "txt_emb_species.npy"
    TXT_NAMES_JSON = "txt_emb_species.json"
    RANKS = ("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
    
    def __init__(self, rank: str = "Class", target_taxon: str = "Animalia Chordata Mammalia", min_confidence: float = 0.3):
        """
        Initialize BioCLIP detector
        Args:
            rank: Taxonomic rank to classify at (default: "Class")
            target_taxon: Target taxonomic group to detect (default: "Animalia Chordata Mammalia")
            min_confidence: Minimum confidence threshold (default: 0.3)
        """
        # Validate rank
        if rank not in self.RANKS:
            raise ValueError(
                f"Invalid rank: {rank}. Must be one of: {', '.join(self.RANKS)}"
            )
        
        self.rank = rank
        self.target_taxon = target_taxon
        self.min_confidence = min_confidence
        
        print(f"BioCLIP detector initialized:")
        print(f"  - Rank: {rank}")
        print(f"  - Target taxon: {target_taxon}")
        print(f"  - Min confidence: {min_confidence}")
        self.model = None
        self.txt_emb = None
        self.txt_names = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        self.load_model()

    def _download_bioclip_data_files(self):
        """Download required BioCLIP data files if needed."""
        # Check multiple possible locations
        possible_paths = [
            ".",  # Current directory
            "/app",  # Docker app directory
            os.getcwd(),  # Current working directory
        ]
        
        # Try to find existing files first
        for base_path in possible_paths:
            npy_path = os.path.join(base_path, self.TXT_EMB_NPY)
            json_path = os.path.join(base_path, self.TXT_NAMES_JSON)
            
            if os.path.exists(npy_path) and os.path.exists(json_path):
                print(f"Found BioCLIP embeddings at: {base_path}")
                return npy_path, json_path
        
        # If not found, try to download (only works if not in offline mode)
        print("BioCLIP text embeddings not found locally, attempting download...")
        repo_id = "imageomics/bioclip-demo"
        
        try:
            npy_path = hf_hub_download(
                repo_id=repo_id, filename=self.TXT_EMB_NPY, repo_type="space",
                local_dir=".", local_dir_use_symlinks=False
            )
            json_path = hf_hub_download(
                repo_id=repo_id, filename=self.TXT_NAMES_JSON, repo_type="space",
                local_dir=".", local_dir_use_symlinks=False
            )
            return npy_path, json_path
        except Exception as e:
            raise FileNotFoundError(
                f"BioCLIP text embeddings not found and download failed: {e}\n"
                f"Please ensure {self.TXT_EMB_NPY} and {self.TXT_NAMES_JSON} "
                f"are present in one of: {possible_paths}"
            )

    def load_model(self):
        """Load BioCLIP model and text embeddings"""
        print("Loading BioCLIP model...")
        
        # Temporarily disable HF_HUB_OFFLINE to allow loading from cache
        # (open_clip needs to check cache via HF Hub, even in offline environments)
        original_offline = os.environ.get('HF_HUB_OFFLINE', None)
        try:
            # Unset HF_HUB_OFFLINE if it's set
            if 'HF_HUB_OFFLINE' in os.environ:
                del os.environ['HF_HUB_OFFLINE']
            
            # Load model - use cache_dir to find pre-downloaded weights
            cache_dir = os.environ.get('HF_HOME', '/hf_cache')
            print(f"Loading BioCLIP from cache: {cache_dir}")
            
            self.model, _, _ = open_clip.create_model_and_transforms(
                'hf-hub:imageomics/bioclip',
                cache_dir=cache_dir
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✓ BioCLIP model loaded successfully")
            
        finally:
            # Restore original HF_HUB_OFFLINE setting
            if original_offline is not None:
                os.environ['HF_HUB_OFFLINE'] = original_offline
        
        # Download and load text embeddings
        npy_path, json_path = self._download_bioclip_data_files()
        self.txt_emb = torch.from_numpy(np.load(npy_path, mmap_mode="r")).to(self.device)
        
        with open(json_path) as fd:
            self.txt_names = json.load(fd)
        
        print(f"✓ BioCLIP loaded with {self.txt_emb.shape[1]} species embeddings")

    def _format_name(self, taxon, common):
        """Format taxon name with optional common name."""
        taxon = " ".join(taxon)
        if not common:
            return taxon
        return f"{taxon} ({common})"

    def _get_spatial_attribution(self, img_tensor, top_idx):
        """
        Generate Grad-CAM style spatial attribution map from layer 9.
        Returns 2D spatial map [H, W] showing which regions contributed to classification.
        """
        # Store activation and gradient
        activation = None
        gradient = None
        
        def get_activation(module, input, output):
            nonlocal activation
            if isinstance(output, tuple):
                output = output[0]
            activation = output.detach()
        
        def get_gradient(module, grad_input, grad_output):
            nonlocal gradient
            if isinstance(grad_output[0], torch.Tensor):
                gradient = grad_output[0].detach()
        
        # Register hooks on layer 9
        hook_handles = []
        for name, module in self.model.named_modules():
            if name == 'visual.transformer.resblocks.9':
                hook_handles.append(module.register_forward_hook(get_activation))
                hook_handles.append(module.register_full_backward_hook(get_gradient))
                break
        
        try:
            # Forward pass
            img_features = self.model.encode_image(img_tensor)
            img_features = F.normalize(img_features, dim=-1)
            
            # Calculate logits
            logits = (self.model.logit_scale.exp() * img_features @ self.txt_emb).squeeze()
            
            # Backward pass
            self.model.zero_grad()
            logits[top_idx].backward(retain_graph=True)
            
            # Generate attribution map
            if activation is not None and gradient is not None:
                # Transformer output: [B, N, C]
                weights = gradient.abs().mean(dim=2)  # [B, N]
                cam = weights  # [B, N]
                
                # Reshape to 2D spatial grid (remove CLS token)
                B, N = cam.shape
                grid_size = int(np.sqrt(N - 1))
                
                if grid_size * grid_size == N - 1:
                    cam_spatial = cam[:, 1:].reshape(B, grid_size, grid_size)
                    cam = cam_spatial.squeeze(0).cpu().numpy()
                    
                    # Normalize
                    if cam.max() > cam.min():
                        cam = (cam - cam.min()) / (cam.max() - cam.min())
                    
                    return cam
            
            return None
            
        finally:
            # Remove hooks
            for handle in hook_handles:
                handle.remove()

    def _get_bboxes_from_heatmap(self, heatmap: np.ndarray, threshold: float = 0.5, 
                                  max_boxes: int = 5) -> list:
        """Extract bounding boxes from attribution heatmap."""
        # Threshold the heatmap
        binary_mask = (heatmap > threshold).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        if num_labels <= 1:
            return []
        
        # Collect all valid components
        bboxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < 10:  # min_area
                continue
            
            # Get bounding box
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate average intensity
            mask = (labels == i).astype(np.uint8)
            avg_intensity = float(np.mean(heatmap[mask == 1]))
            
            bboxes.append((x, y, x + w, y + h, area, avg_intensity))
        
        if not bboxes:
            return []
        
        # Sort by intensity (descending)
        bboxes.sort(key=lambda x: x[5], reverse=True)
        
        return bboxes[:max_boxes]

    def detect(self, image: Image.Image, target_objects: Union[str, List[str]] = None) -> Tuple[List[float], List[List[int]], List[str]]:
        """
        Detect objects using BioCLIP classification + gradient-based localization
        Args:
            image: Input image
            target_objects: Ignored for BioCLIP (uses self.target_taxon)
        Returns:
            Tuple of (rewards, bboxes, labels)
        """
        original_size = image.size
        rank_idx = self.RANKS.index(self.rank)
        
        # Preprocess image
        img_tensor = self.preprocess_img(image).to(self.device).unsqueeze(0)
        img_tensor.requires_grad = True
        
        # Forward pass
        img_features = self.model.encode_image(img_tensor)
        img_features = F.normalize(img_features, dim=-1)
        
        # Calculate logits and probabilities
        logits = (self.model.logit_scale.exp() * img_features @ self.txt_emb).squeeze()
        probs = F.softmax(logits, dim=0)
        
        # Get predictions at specified rank
        if rank_idx + 1 == len(self.RANKS):
            # Species level
            topk = probs.topk(5)
            predictions = {
                self._format_name(*self.txt_names[i]): float(prob) 
                for i, prob in zip(topk.indices, topk.values)
            }
            top_idx = int(topk.indices[0])
        else:
            # Higher rank - aggregate species probabilities
            output = collections.defaultdict(float)
            idx_to_rank = {}
            
            for i in torch.nonzero(probs > 1e-9).squeeze():
                rank_name = " ".join(self.txt_names[i][0][: rank_idx + 1])
                output[rank_name] += probs[i]
                if rank_name not in idx_to_rank:
                    idx_to_rank[rank_name] = []
                idx_to_rank[rank_name].append(i.item())
            
            topk_names = heapq.nlargest(5, output, key=output.get)
            predictions = {name: float(output[name]) for name in topk_names}
            top_rank = topk_names[0]
            top_idx = max(idx_to_rank[top_rank], key=lambda i: probs[i].item())
        
        # Check if target taxon is in predictions with sufficient confidence
        target_found = False
        target_conf = 0.0
        target_label = None
        
        for pred_name, pred_conf in predictions.items():
            if self.target_taxon.lower() in pred_name.lower():
                if pred_conf >= self.min_confidence:
                    target_found = True
                    target_conf = pred_conf
                    target_label = pred_name
                    break
                else:
                    # Found but confidence too low
                    print(f"BioCLIP: Found {pred_name} but confidence {pred_conf:.4f} < {self.min_confidence:.4f}")
                    return [], [], []
        
        if not target_found:
            # Target taxon not detected
            return [], [], []
        
        print(f"BioCLIP detected: {target_label} (confidence: {target_conf:.4f})")
        
        # Get spatial attribution map
        spatial_map = self._get_spatial_attribution(img_tensor, top_idx)
        
        if spatial_map is None:
            return [], [], []
        
        # Resize to 224x224 for bbox extraction
        spatial_map_224 = cv2.resize(spatial_map, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Extract bounding boxes
        bboxes_data = self._get_bboxes_from_heatmap(spatial_map_224, threshold=0.4, max_boxes=5)
        
        if not bboxes_data:
            return [], [], []
        
        # Convert to PTZ app format
        bboxes = []
        labels = []
        rewards = []
        
        scale_x = original_size[0] / 224
        scale_y = original_size[1] / 224
        
        for x1, y1, x2, y2, area, intensity in bboxes_data:
            # Scale bbox to original image size
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            bboxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
            labels.append(target_label)
            
            # reward = 1 - confidence (lower is better)
            reward = 1 - target_conf
            rewards.append(reward)
        
        return rewards, bboxes, labels


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
        
        # BioCLIP can detect any biological organism (doesn't use target_objects)
        if 'bioclip' in model_name.lower():
            return True
            
        # Get valid classes for the model
        valid_classes = {cls.lower() for cls in DetectorFactory.get_model_classes(model_name)}
        
        # Florence can detect any object
        if 'florence' in model_name.lower():
            return True
            
        # Check if any of the target objects are in the valid classes
        return any(obj in valid_classes for obj in target_objects)

    @staticmethod
    def create_detector(
        model_name: str, 
        target_objects: Union[str, List[str]],
        bioclip_rank: str = "Class",
        bioclip_taxon: str = "Animalia Chordata Mammalia",
        bioclip_confidence: float = 0.3
    ) -> 'ObjectDetector':
        """
        Create and return appropriate detector based on model name and validation
        Args:
            model_name: Full model name (e.g., 'yolov8n', 'yolo11n', 'Florence-base', 'yolov8n-oiv7', 'BioCLIP')
            target_objects: Objects to detect (ignored for BioCLIP)
            bioclip_rank: Taxonomic rank for BioCLIP (default: "Class")
            bioclip_taxon: Target taxon for BioCLIP (default: "Animalia Chordata Mammalia")
            bioclip_confidence: Confidence threshold for BioCLIP (default: 0.3)
        """
        model_name_lower = model_name.lower()
        
        # BioCLIP detector
        if 'bioclip' in model_name_lower:
            print(f"Creating BioCLIP detector ({bioclip_rank} rank: {bioclip_taxon}, min confidence: {bioclip_confidence})")
            return BioCLIPDetector(rank=bioclip_rank, target_taxon=bioclip_taxon, min_confidence=bioclip_confidence)
        
        # First validate if the model can detect the objects
        if not DetectorFactory.validate_objects_for_model(model_name, target_objects):
            print(f"Warning: {model_name} cannot detect any of the specified objects.")
            print("Falling back to Florence-base model which has broader detection capabilities.")
            return FlorenceDetector("Florence-base")
            
        # If validation passes, create the requested detector
        yolo_pattern = re.compile(r'^(?:yolov(?:8|9|10)|yolo11)[nsmlex]$')
        yolo_oiv7_pattern = re.compile(r'^yolov8[nsmlex]-oiv7$')
        florence_pattern = re.compile(r'^florence-(base|large)$', re.IGNORECASE)
        
        if 'yolo' in model_name_lower:
            if '-oiv7' in model_name_lower:
                if not yolo_oiv7_pattern.match(model_name_lower):
                    raise ValueError("Invalid YOLO OIV7 model name. Must be: yolov8[n,s,m,l,x]-oiv7")
            elif not yolo_pattern.match(model_name_lower):
                raise ValueError(
                    "Invalid YOLO model name. Must be:\n"
                    "- yolov8[n,s,m,l,x] for YOLOv8\n"
                    "- yolov9[n,s,m,l,x] for YOLOv9\n"
                    "- yolov10[n,s,m,l,x] for YOLOv10\n"
                    "- yolo11[n,s,m,l,x] for YOLOv11"
                )
            return YOLODetector(model_name)
            
        elif 'florence' in model_name_lower:
            if not florence_pattern.match(model_name_lower):
                raise ValueError("Invalid Florence model name. Must be: Florence-base or Florence-large")
            return FlorenceDetector(model_name)
            
        else:
            raise ValueError(
                "Invalid model type. Must be either:\n"
                "- YOLO (e.g., 'yolov8n', 'yolo11n')\n"
                "- YOLO OIV7 (e.g., 'yolov8n-oiv7')\n"
                "- Florence (e.g., 'Florence-base')\n"
                "- BioCLIP (e.g., 'BioCLIP')"
            )

def get_label_from_image_and_object(
    image: Image.Image,
    target_object: str,
    detector: Union[ObjectDetector, List[ObjectDetector]],
    processor=None  # Kept for backwards compatibility
) -> List[Dict]:
    """
    Unified interface for object detection
    Args:
        image: Input image
        target_object: Target object(s) to detect
        detector: Single detector or list of detectors
        processor: Kept for backwards compatibility
    Returns: List of dictionaries with 'reward', 'bbox', 'label', and 'model' keys
    """
    # Handle both single detector and list of detectors
    if isinstance(detector, list):
        # Multiple detectors - run each and combine results
        all_detections = []
        
        for det in detector:
            # Get a human-readable model name
            model_name = _get_model_name(det)
            print(f"Running detection with {model_name}...")
            rewards, bboxes, labels = det.detect(image, target_object)
            
            # Convert to list of dictionaries
            detections = []
            for reward, bbox, label in zip(rewards, bboxes, labels):
                detections.append({
                    'reward': reward,
                    'bbox': bbox,
                    'label': label,
                    'model': model_name
                })
            
            print(f"  Found {len(detections)} detections")
            all_detections.append(detections)
        
        # Combine detections from all models
        combined_results = combine_detections_from_models(all_detections)
        print(f"Combined total: {len(combined_results)} detections after merging")
        return combined_results
    else:
        # Single detector - original behavior
        model_name = _get_model_name(detector)
        rewards, bboxes, labels = detector.detect(image, target_object)
        
        # Convert to list of dictionaries
        results = []
        for reward, bbox, label in zip(rewards, bboxes, labels):
            results.append({
                'reward': reward,
                'bbox': bbox,
                'label': label,
                'model': model_name
            })
        
        if not results:
            return []
            
        return results


def _get_model_name(detector: ObjectDetector) -> str:
    """Get a human-readable model name from a detector instance"""
    if isinstance(detector, YOLODetector):
        return detector.model_name
    elif isinstance(detector, FlorenceDetector):
        return f"Florence-{detector.model_size}"
    elif isinstance(detector, BioCLIPDetector):
        return f"BioCLIP-{detector.rank}"
    else:
        return detector.__class__.__name__
