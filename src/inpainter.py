import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
import numpy as np

class WatermarkInpainter:
    def __init__(self, device=None):
        """
        Initialize the LaMA inpainter.
        :param device: torch device (cuda or cpu).
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Initializing LaMA on {self.device}")
        self.simple_lama = SimpleLama(device=self.device)

    def remove_watermark(self, image, mask):
        """
        Remove watermark using LaMA.
        :param image: PIL Image or numpy array (H, W, 3).
        :param mask: PIL Image or numpy array (H, W) - 255 for mask, 0 for background.
        :return: Clean PIL Image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
            
        # Ensure mask is L mode
        if mask.mode != 'L':
            mask = mask.convert('L')
            
        result = self.simple_lama(image, mask)
        return result
