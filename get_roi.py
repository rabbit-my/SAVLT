import torch
import numpy as np
import cv2
from typing import Optional, Tuple, List
from segmentation_models_pytorch import Unet

class ROIExtractor:
    def __init__(self, model_path: str, device: Optional[str] = None, morph_erode_retention: float = 100.0, morph_dilate_expansion: float = 100.0):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._init_model(model_path).to(self.device)
        self.model.eval()
        self.morph_erode_retention = morph_erode_retention
        self.morph_dilate_expansion = morph_dilate_expansion
        
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
    def dilate_mask_to_expansion(mask: np.ndarray, expansion_rate: float) -> np.ndarray:
    
        if expansion_rate <= 100:
            return mask.copy()
        
        mask_u8 = (mask.astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)
        original_area = np.sum(mask_u8 > 0)
        image_area = mask_u8.shape[0] * mask_u8.shape[1]
        target_area = min(
            original_area * (expansion_rate / 100),
            image_area
        )
        if original_area >= target_area:
            return mask.copy()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = mask_u8.copy()
        for _ in range(500):
            dilated = cv2.dilate(dilated, kernel)
            current_area = np.sum(dilated > 0)
            if current_area >= target_area or current_area >= image_area:
                break
        
        return (dilated > 0).astype(np.uint8)

    @staticmethod
    def erode_mask_to_retention(mask: np.ndarray, retention_rate: float) -> np.ndarray:

        if retention_rate >= 100:
            return mask.copy()
        
        mask_u8 = (mask.astype(np.uint8) * 255) if mask.max() <= 1 else mask.astype(np.uint8)
        original_area = np.sum(mask_u8 > 0)
        target_area = original_area * (retention_rate / 100)
        if target_area <= 0:
            return mask.copy()
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = mask_u8.copy()
        for _ in range(500):
            eroded = cv2.erode(eroded, kernel)
            current_area = np.sum(eroded > 0)
            if current_area <= target_area or current_area == 0:
                break
        
        return (eroded > 0).astype(np.uint8)

    @staticmethod
    def find_largest_foreground_bbox(mask: np.ndarray, min_y_ratio: float = 0.1) -> Optional[Tuple[float, float, float, float]]:

        h, w = mask.shape
        min_y = int(h * min_y_ratio)
        
        for y in range(min_y, h):
            if np.all(mask[y, :] == 1):
                return (0.0, y / h, 1.0, 1.0)
        return None

    @staticmethod
    def mask_to_patch_indices(
        mask: np.ndarray,
        num_patches_per_side: int = 14,
        full_overlap_only: bool = True,
    ) -> List[int]:
        h, w = mask.shape
        patch_h = h // num_patches_per_side
        patch_w = w // num_patches_per_side
        indices = []
        for i in range(num_patches_per_side):
            for j in range(num_patches_per_side):
                y1, y2 = i * patch_h, (i + 1) * patch_h
                x1, x2 = j * patch_w, (j + 1) * patch_w
                region = mask[y1:y2, x1:x2]
                if full_overlap_only:
                    if np.all(region == 1):
                        indices.append(i * num_patches_per_side + j)
                else:
                    if np.any(region == 1):
                        indices.append(i * num_patches_per_side + j)
        return indices

    @staticmethod
    def bbox_to_patch_indices(
        bbox: Tuple[float, float, float, float],
        num_patches_per_side: int = 14,
    ) -> List[int]:
        x_min, y_min, x_max, y_max = bbox
        norm_patch_size = 1.0 / num_patches_per_side
        start_x = int(x_min / norm_patch_size)
        start_y = int(y_min / norm_patch_size)
        end_x = int(x_max / norm_patch_size)
        end_y = int(y_max / norm_patch_size)
        indices = []
        for y in range(start_y, min(end_y + 1, num_patches_per_side)):
            for x in range(start_x, min(end_x + 1, num_patches_per_side)):
                patch_x_min = x * norm_patch_size
                patch_y_min = y * norm_patch_size
                patch_x_max = (x + 1) * norm_patch_size
                patch_y_max = (y + 1) * norm_patch_size
                if (x_min <= patch_x_min and patch_x_max <= x_max and
                    y_min <= patch_y_min and patch_y_max <= y_max):
                    indices.append(y * num_patches_per_side + x)
        return indices
    
    def extract_from_batch(self, image_tensor: torch.Tensor, num_patches_per_side: int = 14) -> List[List[int]]:
        with torch.no_grad():
            masks = self.model(image_tensor.to(self.device))
            pred_masks = (masks.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8)
        
        result = []
        for mask in pred_masks:
            if self.morph_erode_retention < 100:
                mask = self.erode_mask_to_retention(mask, self.morph_erode_retention)
            if self.morph_dilate_expansion > 100:
                mask = self.dilate_mask_to_expansion(mask, self.morph_dilate_expansion)
            indices = self.mask_to_patch_indices(
                mask,
                num_patches_per_side=num_patches_per_side,
                full_overlap_only=True,
            )
            if len(indices) == 0:
                bbox = self.find_largest_foreground_bbox(mask)
                if bbox is not None:
                    indices = self.bbox_to_patch_indices(bbox, num_patches_per_side)
            if len(indices) == 0:
                indices = list(range(num_patches_per_side ** 2))
            result.append(indices)
        return result
