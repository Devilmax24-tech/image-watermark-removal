import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import easyocr

class WatermarkDetector:
    def __init__(self, model_path='models/best.pt'):
        """
        Initialize the YOLOv11 detector and EasyOCR.
        :param model_path: Path to the YOLOv11 weights file.
        """
        if not os.path.exists(model_path):
            print(f"CRITICAL: Model file not found at {model_path}")
            print("Please upload your YOLOv11 'best.pt' weights to the models/ folder.")
            self.model = None
        else:
            self.model = YOLO(model_path)
        
        # Initialize OCR as a fallback for text-only watermarks
        print("  [Detector] Initializing OCR engine...")
        self.reader = easyocr.Reader(['en'], gpu=False)
        
    def detect(self, image_path, conf=0.15):
        """
        Nuclear mode detection with performance optimization.
        1. YOLO first (fastest)
        2. OCR passes (fallback, slow)
        """
        boxes = []
        
        # 0. Prep Image
        if isinstance(image_path, str):
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return []
            img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        else:
            img = image_path

        # 1. YOLO (Logo Detection) - Move to top for speed
        if self.model is not None:
            results = self.model(image_path, conf=conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    boxes.append(box.xyxy[0].cpu().numpy().astype(int))
        
        if boxes:
            print(f"  [Detector] YOLO found {len(boxes)} candidates. Skipping OCR.")
            return boxes

        # 2. Fallback: OCR passes
        def search_in_ocr(ocr_results):
            found_boxes = []
            targets = ["wedme", "megood", "wed", "good", "edme", "dmegoo", "dmeg", "wmg", "edmeg"]
            for (bbox, text, prob) in ocr_results:
                clean_text = text.lower().strip()
                if any(t in clean_text for t in targets):
                    print(f"  [Found Watermark]: '{text}' (Conf: {prob:.4f})")
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    found_boxes.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])
            return found_boxes

        ocr_kwargs = {'text_threshold': 0.1, 'low_text': 0.05, 'link_threshold': 0.2}

        # Sequential OCR passes - EXIT EARLY on first match
        print("  [Detector] YOLO missed. Running fallback OCR passes...")
        
        # Pass 1: Standard
        ocr_results = self.reader.readtext(img, **ocr_kwargs)
        boxes.extend(search_in_ocr(ocr_results))
        if boxes: return boxes

        # Pass 2: Enhanced Contrast
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0) 
        ocr_results = self.reader.readtext(enhanced, **ocr_kwargs)
        boxes.extend(search_in_ocr(ocr_results))
        if boxes: return boxes

        # Pass 3: 1.5x Scale (Faster than 2x)
        h, w = img.shape[:2]
        scaled = cv2.resize(img, (int(w*1.5), int(h*1.5)), interpolation=cv2.INTER_LINEAR)
        ocr_results = self.reader.readtext(scaled, **ocr_kwargs)
        res_boxes = search_in_ocr(ocr_results)
        for b_box in res_boxes:
            boxes.append([int(x/1.5) for x in b_box])
        if boxes: return boxes

        # Pass 4: Targeted Left-Mid scan (Common watermark location)
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = 0, int(w * 0.5)
        target_slice = img[y1:y2, x1:x2]
        ocr_results = self.reader.readtext(target_slice, **ocr_kwargs)
        found_in_slice = search_in_ocr(ocr_results)
        for b in found_in_slice:
            boxes.append([b[0]+x1, b[1]+y1, b[2]+x1, b[3]+y1])
        
        return boxes

    def create_mask(self, image_shape, boxes, padding=30):
        """
        Create a binary mask from bounding boxes with extra safe padding.
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            # Apply padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_shape[1], x2 + padding)
            y2 = min(image_shape[0], y2 + padding)
            
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask
