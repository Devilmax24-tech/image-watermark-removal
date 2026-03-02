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
        use_gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        print(f"  [Detector] OCR GPU usage: {use_gpu}")
        
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
        
        # Performance optimization: Resize for OCR if image is too large
        max_ocr_dim = 1280
        h, w = img.shape[:2]
        if max(h, w) > max_ocr_dim:
            scale = max_ocr_dim / max(h, w)
            ocr_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            scale_recovery = 1.0 / scale
        else:
            ocr_img = img
            scale_recovery = 1.0
            scale = 1.0

        # Pass 1: Standard
        ocr_results = self.reader.readtext(ocr_img, **ocr_kwargs)
        found_boxes = search_in_ocr(ocr_results)
        if found_boxes:
            return [[int(x * scale_recovery) for x in b] for b in found_boxes]

        # Pass 2: Enhanced Contrast
        gray = cv2.cvtColor(ocr_img, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.convertScaleAbs(gray, alpha=2.0, beta=0) 
        ocr_results = self.reader.readtext(enhanced, **ocr_kwargs)
        found_boxes = search_in_ocr(ocr_results)
        if found_boxes:
            return [[int(x * scale_recovery) for x in b] for b in found_boxes]

        # Pass 3: 1.5x Scale (Optional enhancement for small text)
        if max(h, w) < 1500:
            h_s, w_s = ocr_img.shape[:2]
            scaled = cv2.resize(ocr_img, (int(w_s*1.5), int(h_s*1.5)), interpolation=cv2.INTER_LINEAR)
            ocr_results = self.reader.readtext(scaled, **ocr_kwargs)
            found_boxes = search_in_ocr(ocr_results)
            if found_boxes:
                return [[int(x * scale_recovery / 1.5) for x in b] for b in found_boxes]

        # Pass 4: Targeted scans (Common watermark regions)
        # 1. Left-Mid, 2. Bottom-Center/Right
        regions = [
            (int(h * 0.2), int(h * 0.8), 0, int(w * 0.5)),    # Left-Mid
            (int(h * 0.7), h, int(w * 0.3), w)              # Bottom-Right/Center
        ]
        
        for ry1, ry2, rx1, rx2 in regions:
            # Crop using scaled coordinates for OCR
            y1_s, y2_s = int(ry1 * scale), int(ry2 * scale)
            x1_s, x2_s = int(rx1 * scale), int(rx2 * scale)
            target_slice = ocr_img[y1_s:y2_s, x1_s:x2_s]
            
            if target_slice.size > 0:
                ocr_results = self.reader.readtext(target_slice, **ocr_kwargs)
                found_boxes = search_in_ocr(ocr_results)
                if found_boxes:
                    final_boxes = []
                    for b in found_boxes:
                        # Recover original image coordinates
                        final_boxes.append([
                            int((b[0] + x1_s) * scale_recovery),
                            int((b[1] + y1_s) * scale_recovery),
                            int((b[2] + x1_s) * scale_recovery),
                            int((b[3] + y1_s) * scale_recovery)
                        ])
                    return final_boxes
        
        return []

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
