import torch
import numpy as np
from typing import Optional, Tuple, List
from segmentation_models_pytorch import Unet

class ROIExtractor:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model(model_path).to(self.device)
        self.model.eval()
        
    def _init_model(self, model_path: str) -> torch.nn.Module:
        model = Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model
    
    @staticmethod
    def find_largest_foreground_bbox(mask: np.ndarray, min_y_ratio: float = 0.1) -> Optional[Tuple[float, float, float, float]]:
        h, w = mask.shape
        min_y = int(h * min_y_ratio)
        for y in range(min_y, h):
            if np.all(mask[y, :] == 1):
                return (0.0, y / h, 1.0, 1.0)
        return None
    
    def extract_from_batch(self, image_tensor: torch.Tensor) -> List[Optional[Tuple[float, float, float, float]]]:
        with torch.no_grad():
            masks = self.model(image_tensor.to(self.device))
            pred_masks = (masks.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
            
        return [self.find_largest_foreground_bbox(mask) for mask in pred_masks]
