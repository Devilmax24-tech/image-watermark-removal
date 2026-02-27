import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(original, processed):
    """
    Calculate SSIM and PSNR between original and processed images.
    Note: Since we only change watermark area, we focus on the whole image 
    to ensure background remains identical.
    """
    # Ensure background is the same type and size
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Calculate SSIM
    s = ssim(original, processed, multichannel=True, channel_axis=2)
    
    # Calculate PSNR
    p = psnr(original, processed)
    
    return s, p

def save_image_with_quality(image, path, quality=100):
    """
    Save image preserving quality.
    """
    if isinstance(image, Image.Image):
        image.save(path, quality=quality, subsampling=0)
    else:
        # cv2 uses 0-100 for JPEG quality
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

from PIL import Image
