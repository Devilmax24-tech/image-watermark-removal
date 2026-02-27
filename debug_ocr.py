import easyocr
import cv2
import os

def test_ocr():
    reader = easyocr.Reader(['en'], gpu=False)
    input_dir = 'data/input'
    
    for filename in sorted(os.listdir(input_dir)):
        if "image-1" in filename:
            path = os.path.join(input_dir, filename)
            print(f"\nScanning {filename}...")
            img = cv2.imread(path)
            
            # Pass 1: Normal
            results = reader.readtext(img)
            print("--- Pass 1 (Normal) ---")
            for (bbox, text, prob) in results:
                print(f"  - [{prob:.4f}] '{text}'")
            
            # Pass 2: Enhanced
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            results = reader.readtext(enhanced)
            print("--- Pass 2 (Enhanced) ---")
            for (bbox, text, prob) in results:
                print(f"  - [{prob:.4f}] '{text}'")
            
            # Pass 3: 2x
            h, w = img.shape[:2]
            scaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            results = reader.readtext(scaled)
            print("--- Pass 3 (2x) ---")
            for (bbox, text, prob) in results:
                print(f"  - [{prob:.4f}] '{text}'")

if __name__ == "__main__":
    test_ocr()
