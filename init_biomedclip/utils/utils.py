import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from open_clip import create_model_from_pretrained, get_tokenizer
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
from get_roi import ROIExtractor



def load_clip_model(model_name: str):
    if model_name in ['open_clip-b-16', 'biomedclip-b-16', 'open_clip-l-14']:

        if model_name == 'biomedclip-b-16':
            print("use biomedclip model")
            hf_id = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'

        elif model_name == 'open_clip-b-16':
            hf_id = 'hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K'

        elif model_name == 'open_clip-l-14':
            hf_id = 'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'

        model, preprocess = create_model_from_pretrained(hf_id)
        tokenizer = get_tokenizer(hf_id)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model, preprocess, tokenizer

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


def get_logit_scale(model):
    """获取模型的 logit scale，兼容 PEFT 包装后的模型"""
    m = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if hasattr(m, "logit_scale"):
        return m.logit_scale.exp()
    # 如果模型没有 logit_scale 属性，返回默认值（通常 CLIP 模型的 logit_scale.exp() 约为 100）
    device = next(model.parameters()).device if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else torch.device('cpu')
    return torch.tensor(100.0, device=device)



@torch.no_grad()
def evaluate_lora(
    args,
    clip_model,
    test_loader,
    text_features,
    visual_extractor,
    prototypes,
    roi_extractor=None,
    tokenizer=None,
    texts=None
):
    """
    评估函数
    
    参数:
        text_features: 文本特征（如果文本编码器可训练，可能为None，需要重新计算）
        tokenizer: tokenizer（仅在文本编码器可训练时需要）
        texts: 文本列表（仅在文本编码器可训练时需要）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.eval()
    visual_extractor.eval()
    
    # 如果文本编码器可训练，需要重新计算 text_features
    if args.use_text_peft and tokenizer is not None and texts is not None:
        from tuning_model import extract_text_features
        text_features = extract_text_features(clip_model, tokenizer, texts, device)

    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_binary_preds = []
    all_binary_labels = []
    all_logits = []
    # 如果未传入 roi_extractor，则创建一个（向后兼容）
    if roi_extractor is None:
        roi_extractor = ROIExtractor(model_path=args.seg_path)

    for images, labels in tqdm.tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        rois = roi_extractor.extract_from_batch(images)
        
        image_features = visual_extractor(images, rois)
        
        scale = get_logit_scale(clip_model)
        logits_text = scale * (image_features @ text_features.T)
        logits_visual = scale * (image_features @ prototypes().T)
        
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
