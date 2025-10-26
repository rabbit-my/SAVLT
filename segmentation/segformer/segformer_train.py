import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import KFold
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.metrics import iou_score, f1_score
from segmentation_models_pytorch.losses import DiceLoss
import cv2
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.metrics import get_stats, iou_score
import logging
import sys
from segmentation_models_pytorch import Segformer


log_path = "segformer_train_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),         
        logging.StreamHandler(sys.stdout)      
    ]
)

print = logging.info


def compute_metrics(outputs, masks):
    if masks.ndim == 3:  # [B, H, W]
        masks = masks.unsqueeze(1)  
    
    masks_int = (masks > 0.5).long()  
    tp, fp, fn, tn = get_stats(outputs, masks_int, mode='binary', threshold=0.5)
    iou = iou_score(tp, fp, fn, tn, reduction='micro')
    dice = f1_score(tp, fp, fn, tn, reduction='micro')
    return iou, dice



class Config:
    image_base = "/home/dataset/OCTseg/images/"
    mask_base = "/home/dataset/OCTseg/binary_mask/"
    categories = ["", "", ""]
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    num_epochs = 50
    lr = 1e-4
    num_workers = 4
    encoder_name = 'mit_b0'
    encoder_weights = 'imagenet'
    n_splits = 5  
    image_size = 224  

config = Config()

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(
            size=(config.image_size, config.image_size),  
            scale=(0.75, 1.0),
            interpolation=cv2.INTER_CUBIC
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.RandomResizedCrop(
            size=(config.image_size, config.image_size), 
            scale=(0.75, 1.0),
            interpolation=cv2.INTER_CUBIC
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32) 
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(np.expand_dims(mask, axis=0)).float()
        
        return {
            'image': image,
            'mask': mask
        }

def get_all_paths():
    image_paths = []
    mask_paths = []
    
    for category in config.categories:
        category_image_dir = os.path.join(config.image_base, category)
        category_mask_dir = os.path.join(config.mask_base, category)
        
        for image_name in os.listdir(category_image_dir):
            image_path = os.path.join(category_image_dir, image_name)
            mask_path = os.path.join(category_mask_dir, image_name)
            
            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)
            else:
                print(f"Warning: Mask not found for {image_path}")
    
    return image_paths, mask_paths



def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    
    for data in tqdm(loader, desc="Training"):
        inputs = data['image'].to(device)
        masks = data['mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        
        iou, dice = compute_metrics(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_iou += iou.item() * inputs.size(0)
        running_dice += dice.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_iou = running_iou / len(loader.dataset)
    epoch_dice = running_dice / len(loader.dataset)
    
    return epoch_loss, epoch_iou, epoch_dice


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            inputs = data['image'].to(device)
            masks = data['mask'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            iou, dice = compute_metrics(outputs, masks)

            
            running_loss += loss.item() * inputs.size(0)
            running_iou += iou.item() * inputs.size(0)
            running_dice += dice.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_iou = running_iou / len(loader.dataset)
    epoch_dice = running_dice / len(loader.dataset)
    
    return epoch_loss, epoch_iou, epoch_dice

def train_model():
    image_paths, mask_paths = get_all_paths()
    
    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    
    fold_results = defaultdict(list)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths)):
        print(f"\n{'='*30} Fold {fold + 1}/{config.n_splits} {'='*30}")
        
        train_image_paths = [image_paths[i] for i in train_idx]
        train_mask_paths = [mask_paths[i] for i in train_idx]
        val_image_paths = [image_paths[i] for i in val_idx]
        val_mask_paths = [mask_paths[i] for i in val_idx]
        
        train_dataset = SegmentationDataset(
            train_image_paths, 
            train_mask_paths, 
            transform=get_train_transforms()
        )
        val_dataset = SegmentationDataset(
            val_image_paths, 
            val_mask_paths, 
            transform=get_val_transforms()
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers
        )
        
        model = Segformer(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            encoder_depth=5,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        ).to(config.device)
        
        criterion = DiceLoss(mode='binary')
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        
        best_val_dice = 0.0
        
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            
            train_loss, train_iou, train_dice = train_epoch(
                model, train_loader, optimizer, criterion, config.device
            )
            
            val_loss, val_iou, val_dice = validate_epoch(
                model, val_loader, criterion, config.device
            )
            
            scheduler.step(val_loss)
            
            print(
                f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train Dice: {train_dice:.4f}\n"
                f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}"
            )
            
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(model.state_dict(), f"segformer_best_model_fold{fold}.pth")
                print(f"New best model saved with Dice: {best_val_dice:.4f}")
            
            fold_results['fold'].append(fold)
            fold_results['epoch'].append(epoch)
            fold_results['train_loss'].append(train_loss)
            fold_results['train_iou'].append(train_iou)
            fold_results['train_dice'].append(train_dice)
            fold_results['val_loss'].append(val_loss)
            fold_results['val_iou'].append(val_iou)
            fold_results['val_dice'].append(val_dice)
    
    print("\nCross-validation results summary:")
    for fold in range(config.n_splits):
        fold_idx = [i for i, f in enumerate(fold_results['fold']) if f == fold]
        best_epoch = np.argmax([fold_results['val_dice'][i] for i in fold_idx])
        
        print(f"\nFold {fold + 1} Best Results:")
        print(f"Epoch: {best_epoch + 1}")
        print(f"Train Loss: {fold_results['train_loss'][fold_idx[best_epoch]]:.4f}")
        print(f"Train IoU: {fold_results['train_iou'][fold_idx[best_epoch]]:.4f}")
        print(f"Train Dice: {fold_results['train_dice'][fold_idx[best_epoch]]:.4f}")
        print(f"Val Loss: {fold_results['val_loss'][fold_idx[best_epoch]]:.4f}")
        print(f"Val IoU: {fold_results['val_iou'][fold_idx[best_epoch]]:.4f}")
        print(f"Val Dice: {fold_results['val_dice'][fold_idx[best_epoch]]:.4f}")

if __name__ == "__main__":
    train_model()
