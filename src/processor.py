import os
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from src.detector import WatermarkDetector
from src.inpainter import WatermarkInpainter
from src.utils import calculate_metrics, save_image_with_quality

class BatchProcessor:
    def __init__(self, model_path='models/best.pt', num_workers=4, checkpoint_file='checkpoint.txt'):
        self.detector = WatermarkDetector(model_path)
        self.inpainter = WatermarkInpainter()
        self.num_workers = num_workers
        self.checkpoint_file = checkpoint_file
        self.processed_files = self._load_checkpoints()
        self.stats = {
            'processed': len(self.processed_files),
            'failed': 0,
            'skipped': 0,
            'start_time': 0
        }
        self.lock = threading.Lock()

    def _load_checkpoints(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def _save_checkpoint(self, filename):
        with self.lock:
            with open(self.checkpoint_file, 'a') as f:
                f.write(f"{filename}\n")
            self.processed_files.add(filename)

    def process_single_image(self, image_path, output_dir):
        """
        Process a single image: detect -> mask -> inpaint -> save.
        """
        filename = os.path.basename(image_path)
        print(f"--- Processing: {filename} ---")
        
        if filename in self.processed_files:
            with self.lock:
                self.stats['skipped'] += 1
            return True

        try:
            output_path = os.path.join(output_dir, filename)
            
            # Load image
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # 1. Detect
            boxes = self.detector.detect(img_rgb)
            
            if not boxes:
                # No watermark detected, save original
                save_image_with_quality(img_rgb, output_path)
                self._save_checkpoint(filename)
                with self.lock:
                    self.stats['processed'] += 1
                return True
            
            # 2. Create Mask
            mask = self.detector.create_mask(img_rgb.shape, boxes)
            mask_path = os.path.join("data/masks", filename)
            cv2.imwrite(mask_path, mask)
            
            # 3. Inpaint
            inpainted_pil = self.inpainter.remove_watermark(img_rgb, mask)
            inpainted_np = np.array(inpainted_pil)
            
            # 4. Blend: Only take pixels where the mask is active
            # This ensures 100% of the original background is preserved
            final_img_np = img_rgb.copy()
            mask_3ch = np.stack([mask]*3, axis=-1) / 255.0
            
            # Ensure inpainted_np is the same size as img_rgb
            if inpainted_np.shape != img_rgb.shape:
                inpainted_np = cv2.resize(inpainted_np, (img_rgb.shape[1], img_rgb.shape[0]))
                
            final_img_np = (img_rgb * (1 - mask_3ch) + inpainted_np * mask_3ch).astype(np.uint8)
            
            # 5. Validate Quality
            ssim_val, psnr_val = calculate_metrics(img_rgb, final_img_np)
            
            # 6. Save
            final_img_pil = Image.fromarray(final_img_np)
            save_image_with_quality(final_img_pil, output_path)
            
            self._save_checkpoint(filename)
            print(f"Processed {filename}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}dB")
            
            with self.lock:
                self.stats['processed'] += 1
            return True

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            with self.lock:
                self.stats['failed'] += 1
            return False

    def process_batch(self, image_paths, output_dir):
        """
        Process a list of images using multi-threading.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.stats['start_time'] = time.time()
        self.stats['total'] = len(image_paths)
        
        initial_processed = len(self.processed_files)
        print(f"Skipping {initial_processed} already processed images.")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            list(executor.map(lambda p: self.process_single_image(p, output_dir), image_paths))
            
        end_time = time.time()
        duration = end_time - self.stats['start_time']
        actually_processed = self.stats['processed'] - initial_processed
        speed = (actually_processed / duration) * 60 if duration > 0 else 0
        
        print("\n--- Processing Summary ---")
        print(f"Total Images: {self.stats['total']}")
        print(f"Already Processed (Skipped): {self.stats['skipped']}")
        print(f"Newly Processed: {actually_processed}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Total Time: {duration:.2f} seconds")
        print(f"Average Speed: {speed:.2f} images/minute")
        print("--------------------------\n")
