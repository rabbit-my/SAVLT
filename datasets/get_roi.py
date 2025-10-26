import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet


def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class ForegroundBBoxExtractor:
    def __init__(self, pth_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        ).to(self.device)
        self.model.load_state_dict(torch.load(pth_path, map_location=self.device))
        self.model.eval()
        self.transform = get_val_transforms()

    def predict_mask(self, impath):
        image = cv2.imread(impath)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        return pred_mask

def find_largest_foreground_bbox(mask, min_y_ratio=0.1):
    h, w = mask.shape
    min_y = int(h * min_y_ratio)

    for y in range(min_y, h):
        if np.all(mask[y, :] == 1):
            return (0.0, y / h, 1.0, 1.0)  # x1, y1, x2, y2（归一化）
    
    return None  
