import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
from get_roi import ROIExtractor
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
import logging
from datetime import datetime
import os


def load_clip_model(model_name: str):
    """
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-large-patch14-336"
    """
    
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def count_parameters(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"可训练参数: {param.shape} | 数量: {param.numel()}")
    return {
        "total_params": all_params,
        "trainable_params": trainable_params,
        "trainable_percentage": 100 * trainable_params / all_params
    }
    
def format_num_params(num):
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    return str(num)


@torch.no_grad()
def evaluate_lora(
    args,
    clip_model,
    test_loader,
    text_features,
    visual_extractor,
    prototypes
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.eval()
    visual_extractor.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_binary_preds = []
    all_binary_labels = []
    all_logits = []
    roi_extractor = ROIExtractor(model_path=args.seg_path)

    for images, labels in tqdm.tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        rois = roi_extractor.extract_from_batch(images)
        
        image_features = visual_extractor(images, rois)
        # logits = clip_model.logit_scale.exp() * image_features @ text_features.T
        
        
        # logits_text = clip_model.logit_scale.exp() * image_features @ text_features.T
        # logits_visual = clip_model.logit_scale.exp() * image_features @ prototypes().T
        
        
        logits_text = 100 * image_features @ text_features.T
        logits_visual = 100 * image_features @ prototypes().T
        
        logits = (1- args.proto_alpha) * logits_text + args.proto_alpha * logits_visual
        
        preds = torch.argmax(logits, dim=1)


        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_logits.append(logits.cpu())

        binary_labels = torch.tensor([1 if l in [0, 3] else 0 for l in labels], device=labels.device)
        binary_preds = torch.tensor([1 if p in [0, 3] else 0 for p in preds], device=preds.device)

        all_binary_labels.extend(binary_labels.cpu().tolist())
        all_binary_preds.extend(binary_preds.cpu().tolist())

    acc = correct / total * 100

    binary_correct = sum([1 for p, l in zip(all_binary_preds, all_binary_labels) if p == l])
    binary_total = len(all_binary_labels)
    binary_acc = binary_correct / binary_total * 100

    all_logits = torch.cat(all_logits, dim=0) 

    pos_score = all_logits[:, [0, 3]].sum(dim=1) 
    neg_score = all_logits[:, [1, 2, 4]].sum(dim=1) 

    binary_probs = torch.stack([neg_score, pos_score], dim=1)
    binary_softmax = torch.softmax(binary_probs, dim=1)[:, 1]  

    auc = roc_auc_score(all_binary_labels, binary_softmax.numpy())
    cm = confusion_matrix(all_binary_labels, all_binary_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)
    ppv = tp / (tp + fp + 1e-6)
    npv = tn / (tn + fn + 1e-6)
    return {
        "acc_5class": acc,
        "acc_binary": binary_acc,
        "auc_binary": auc * 100,
        "sensitivity": sensitivity * 100,
        "specificity": specificity * 100,
        "ppv": ppv * 100,
        "npv": npv * 100
    }


def setup_logger(save_dir):
    import logging
    import os

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "log_test.txt")

    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear() 

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    return log_file

