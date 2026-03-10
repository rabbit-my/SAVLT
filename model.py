from typing import List,Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import evaluate_lora,load_clip_model
from utils.prompt import descriptions
import torch
from get_roi import ROIExtractor
from peft_model import apply_peft_to_model
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from PIL import Image
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.error_study import export_error_study_lora


def extract_text_features_train(model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    text_features = model.get_text_features(**inputs)
    return text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path, title="Confusion Matrix"):

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 12})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def run_lora(
    args,
    clip_model,
    processor,
    dataset,
    train_loader,
    val_loader,
    test_loader,
    xiangya_test_loader,
    huaxi_test_loader,
    xiangya_error_loader=None,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    texts = [descriptions[classname] for classname in dataset.classnames]
    print(texts)
    print("Applying PEFT to model...")
    base_clip_model = clip_model
    
    vision_lora_config = json.loads(args.vision_lora_config)
    
    clip_model = apply_peft_to_model(
        base_clip_model,
        model_name= args.model_name,
        peft_type = args.peft_type,
        use_vision_peft =args.use_vision_peft,
        use_text_peft =args.use_text_peft,
        vision_config_kwargs=vision_lora_config,
        text_config_kwargs=args.text_lora_config
    ).to(device)
    
    
    visual_extractor = ROIAttentionExtractor(clip_model).to(device)

    num_classes = len(dataset.classnames)

    prototypes = VisualPrototypes(num_classes, clip_model.config.projection_dim).to(device)
    print(prototypes().shape) # [5 * 512]
    params = list(filter(lambda p: p.requires_grad, clip_model.parameters())) + \
         list(visual_extractor.parameters())
    params += list(prototypes.parameters())
    
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_iters * args.shots, eta_min=1e-6)
    print("proto is learnable?")
    print([p.requires_grad for p in prototypes.parameters()])

    roi_extractor = ROIExtractor(
        model_path=args.seg_path,
        morph_erode_retention=getattr(args, 'morph_erode_retention', 100.0),
        morph_dilate_expansion=getattr(args, 'morph_dilate_expansion', 100.0),
    )
    count_iters = 0
    total_iters = args.n_iters * args.shots
    clip_model.train()


    while count_iters < total_iters:
        for (images, labels) in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            text_features = extract_text_features_train(clip_model, processor, texts, device=device)
            optimizer.zero_grad()
            rois = roi_extractor.extract_from_batch(images)
            image_features= visual_extractor(images, rois)
            
            logits_text = clip_model.logit_scale.exp() * image_features @ text_features.T
            logits_visual = clip_model.logit_scale.exp() * image_features @ prototypes().T
            
            loss_text = F.cross_entropy(logits_text, labels)
            loss_visual = F.cross_entropy(logits_visual, labels)

            loss =(1- args.proto_alpha) *  loss_text + args.proto_alpha * loss_visual

            loss.backward()
                
            optimizer.step()
            scheduler.step()

            count_iters += 1
            if count_iters >= total_iters:
                break

        # print(f"[Iter {count_iters}/{total_iters}] Loss: {loss.item():.4f}")
        
        # if you do not need val resullt ,you can explanatory note follows.
        print_interval = 5
        if count_iters % print_interval == 0 or count_iters >= total_iters:
            val_results = evaluate_lora(
                args, clip_model, val_loader,
                text_features, visual_extractor,prototypes
            )
            acc_val = val_results.get("acc_5class", 0.0)
            # print(f"[Iter {count_iters}/{total_iters}] Loss: {loss.item():.4f} | Val Acc: {acc_val:.2f}%")
            print(f"[Iter {count_iters}/{total_iters}] "
                f"Total Loss: {loss.item():.4f} | "
                f"Text Loss: {loss_text.item():.4f} | "
                f"Proto Loss: {loss_visual.item():.4f} | "
                f"Val Acc: {acc_val:.2f}%")


    multicenter_results = evaluate_lora(args, clip_model, test_loader, text_features, visual_extractor, prototypes)
    xiangya_results = evaluate_lora(args, clip_model, xiangya_test_loader, text_features, visual_extractor, prototypes)
    huaxi_results = evaluate_lora(args, clip_model, huaxi_test_loader, text_features, visual_extractor, prototypes)
    
    print_keys = ["acc_5class", "acc_binary", "auc_binary", "sensitivity", "specificity", "ppv", "npv"]
    print("\nMulticenter Results:", {k: f"{v:.2f}%" for k, v in multicenter_results.items() if k in print_keys})
    print("Xiangya Results:", {k: f"{v:.2f}%" for k, v in xiangya_results.items() if k in print_keys})
    print("Huaxi Results:", {k: f"{v:.2f}%" for k, v in huaxi_results.items() if k in print_keys})

    cm_save_dir = os.path.join("confusion_matrices", f"seed_{args.seed}")
    os.makedirs(cm_save_dir, exist_ok=True)
    
    class_names = dataset.classnames
    
    plot_confusion_matrix(
        multicenter_results["all_labels"], 
        multicenter_results["all_preds"],
        class_names,
        os.path.join(cm_save_dir, "multicenter_confusion_matrix.png"),
        title=f"Multicenter Test Set (Seed={args.seed})"
    )
    
    plot_confusion_matrix(
        xiangya_results["all_labels"], 
        xiangya_results["all_preds"],
        class_names,
        os.path.join(cm_save_dir, "xiangya_confusion_matrix.png"),
        title=f"Xiangya Test Set (Seed={args.seed})"
    )
    
    plot_confusion_matrix(
        huaxi_results["all_labels"], 
        huaxi_results["all_preds"],
        class_names,
        os.path.join(cm_save_dir, "huaxi_confusion_matrix.png"),
        title=f"Huaxi Test Set (Seed={args.seed})"
    )
    
    print(f"\n三个测试集的混淆矩阵已保存到: {cm_save_dir}")

    result_dict = {
        "params": {k: v for k, v in vars(args).items()}, 
        "multicenter_results": {k: f"{v:.2f}%" for k, v in multicenter_results.items() if k in print_keys},
        "xiangya_results": {k: f"{v:.2f}%" for k, v in xiangya_results.items() if k in print_keys},
        "huaxi_results": {k: f"{v:.2f}%" for k, v in huaxi_results.items() if k in print_keys},
    }

    print("[RESULT]", json.dumps(result_dict))

    if getattr(args, "error_study", False):
        if xiangya_error_loader is None:
            print("[ErrorStudy] skipped: xiangya_error_loader is None")
        else:
            export_info = export_error_study_lora(
                args=args,
                clip_model=clip_model,
                data_loader_with_paths=xiangya_error_loader,
                classnames=dataset.classnames,
                text_features=text_features,
                visual_extractor=visual_extractor,
                prototypes=prototypes,
                save_root_dir=args.error_study_dir,
                dataset_tag=str(getattr(args, "test_dataset_a", "xiangya")),
                max_samples=int(getattr(args, "error_study_max_samples", 100)),
                include_correct=bool(getattr(args, "error_study_include_correct", False)),
            )
            print(f"[ErrorStudy] saved: {export_info}")
    

class ROIAttentionExtractor(nn.Module):
    def __init__(self, clip_model: nn.Module, query_dim: int = 768, save_dir: str = './attn_vis'):
        super().__init__()
        self.clip_model = clip_model
        self.query_dim = query_dim
        self.save_dir = save_dir

        self.roi_query = nn.Parameter(torch.empty(1, 1, query_dim))
        nn.init.xavier_uniform_(self.roi_query)

    def _parse_roi_indices(self, roi_item, num_patches_per_side: int, total_patches: int) -> List[int]:
        if isinstance(roi_item, (list, tuple)) and len(roi_item) > 0:
            if isinstance(roi_item[0], int):
                return [idx for idx in roi_item if 0 <= idx < total_patches]
            x_min, y_min, x_max, y_max = roi_item
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
            return [idx for idx in indices if 0 <= idx < total_patches]
        return []

    def _forward_vision_with_roi_zeroing(self, pixel_values: torch.Tensor, roi_indices_per_sample: List[List[int]], vision_model, device) -> torch.Tensor:
        """
        策略2：层间对非ROI token置零。
        序列: [CLS, patch_0, ..., patch_195]，patch索引0-195对应序列位置1-196。
        """
        batch_size = pixel_values.size(0)
        seq_len = 197

        hidden_states = vision_model.embeddings(pixel_values)
        hidden_states = vision_model.pre_layrnorm(hidden_states)

        roi_keep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            roi_keep_mask[i, 0] = True
            for idx in roi_indices_per_sample[i]:
                roi_keep_mask[i, idx + 1] = True

        for layer in vision_model.encoder.layers:
            layer_outputs = layer(hidden_states, None, None, output_attentions=False)
            hidden_states = layer_outputs[0]
            hidden_states = hidden_states * roi_keep_mask.unsqueeze(-1).float()

        hidden_states = vision_model.post_layernorm(hidden_states)
        return hidden_states

    def forward(
        self,
        images: torch.Tensor,
        rois_list: List,
        return_attn: bool = False,
        image_names=None,
    ):
        """
        rois_list: List[List[int]] 或 List[Tuple[float,float,float,float]]
        - List[List[int]]: 每张图对应的 patch 索引列表（由 get_roi.mask_to_patch_indices 得到）
        - List[Tuple]: 兼容旧的 bbox 格式 (x_min, y_min, x_max, y_max)
        """
        device = next(self.parameters()).device

        if not isinstance(images, torch.Tensor):
            raise TypeError("images 应该是 Tensor 格式，请确认图像是否经过 transform。")
        if images.dim() != 4:
            raise ValueError(f"images 应该是 [B, 3, H, W] 维度，但现在是 {images.shape}")

        batch_size = images.size(0)
        if len(rois_list) != batch_size:
            raise ValueError(f"ROI数量({len(rois_list)})与图像数量({batch_size})不匹配")

        pixel_values = images.to(device)

        vision_model = self.clip_model.vision_model
        config = vision_model.config
        patch_size = config.patch_size
        image_size = config.image_size
        num_patches_per_side = image_size // patch_size
        total_patches = num_patches_per_side ** 2

        roi_indices_per_sample = [
            self._parse_roi_indices(rois_list[i], num_patches_per_side, total_patches)
            for i in range(batch_size)
        ]

        last_hidden = self._forward_vision_with_roi_zeroing(pixel_values, roi_indices_per_sample, vision_model, device)
        patch_tokens = last_hidden[:, 1:, :]

        roi_features = []
        roi_indices_all = []
        attn_weights_all = []
        batch_patch_counts = []

        for i in range(batch_size):
            roi_indices = roi_indices_per_sample[i]
            roi_tokens = [patch_tokens[i, idx] for idx in roi_indices]

            if not roi_tokens:
                raise ValueError(f"第{i}张图像的ROI {rois_list[i]}未包含任何有效patch")

            batch_patch_counts.append(len(roi_indices))
            roi_tokens = torch.stack(roi_tokens, dim=0).unsqueeze(0)
            attn_scores = torch.matmul(self.roi_query, roi_tokens.transpose(-1, -2)) / (self.query_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)

            attended = torch.matmul(attn_weights, roi_tokens)
            feat = self.clip_model.visual_projection(attended.squeeze(0).squeeze(0))
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)

            roi_features.append(feat)

            if return_attn:
                roi_indices_all.append(roi_indices)
                attn_weights_all.append(attn_weights.view(-1).detach().cpu())

                if self.save_dir:
                    # Ensure attention map filenames contain original image name
                    if image_names is not None and i < len(image_names) and image_names[i]:
                        # Avoid nested directories / illegal separators
                        full = str(image_names[i])
                        base = os.path.basename(full)
                        parent = os.path.basename(os.path.dirname(full))
                        safe_base = base.replace(os.sep, "_").replace("/", "_")
                        safe_parent = parent.replace(os.sep, "_").replace("/", "_")
                        save_name = f"{safe_parent}__{safe_base}__attn.png"
                    else:
                        save_name = f"attn_map_{i}.png"

                    self.visualize_attention_map(
                        image=images[i].detach().cpu(),
                        attn_weights=attn_weights.view(-1),
                        roi_token_indices=roi_indices,
                        num_patches_per_side=num_patches_per_side,
                        save_path=os.path.join(self.save_dir, save_name)
                    )

        print(f"[ROI] patch tokens per image: {batch_patch_counts} | min={min(batch_patch_counts)}, max={max(batch_patch_counts)}, mean={sum(batch_patch_counts)/len(batch_patch_counts):.1f}")
        roi_features = torch.stack(roi_features, dim=0)
        
        return roi_features
    
    @staticmethod
    def visualize_attention_map(
        image: torch.Tensor,
        attn_weights: torch.Tensor,
        roi_token_indices: List[int],
        num_patches_per_side: int,
        save_path: str,
        alpha: float = 0.7,
        top_k: int = 40  
    ):
        weighted_indices = list(zip(attn_weights.detach().cpu().numpy(), roi_token_indices))
        weighted_indices.sort(reverse=True, key=lambda x: x[0])
        
        full_attn_map = np.zeros((num_patches_per_side, num_patches_per_side))
        
        for weight, idx in weighted_indices[:top_k]:
            y = idx // num_patches_per_side
            x = idx % num_patches_per_side
            full_attn_map[y, x] = weight 
        
        heatmap = Image.fromarray(full_attn_map.astype(np.float32)).resize(
            (image.shape[2], image.shape[1]),
            resample=Image.BICUBIC
        )
        heatmap = np.array(heatmap, dtype=np.float32)
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path.replace(".png", "_orig.png"), image_np)

        cmap = plt.get_cmap("viridis")

        heatmap_rgb = cmap(heatmap)[..., :3]
        overlay = (1 - alpha) * image_np + alpha * heatmap_rgb

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

class VisualPrototypes(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        init_proto = torch.empty(num_classes, feat_dim)
        nn.init.orthogonal_(init_proto)  
        self.prototypes = nn.Parameter(F.normalize(init_proto, p=2, dim=1))

    def forward(self):
        return F.normalize(self.prototypes, dim=1)


if __name__ == '__main__':
    
    model_name = "openai/clip-vit-base-patch16"
    base_model, processor = load_clip_model(model_name)

