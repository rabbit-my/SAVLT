from typing import List,Tuple
import torch.nn.functional as F
from utils.utils import evaluate_lora,load_clip_model
from utils.prompt import descriptions
import torch
from get_roi import ROIExtractor
from peft_model import apply_peft_to_model
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
from PIL import Image
import json
from typing import List, Tuple, Optional


def extract_text_features_train(model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    text_features = model.get_text_features(**inputs)
    return text_features / (text_features.norm(dim=1, keepdim=True) + 1e-6)


def run_lora(args, clip_model, processor, dataset, train_loader, val_loader, test_loader,xiangya_test_loader,huaxi_test_loader):

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
    
    
    visual_extractor = ROIAttentionExtractor(clip_model, num_heads=4).to(device)
    # visual_extractor = ROIAttentionExtractor(base_clip_model).to(device)

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

    roi_extractor = ROIExtractor(model_path= args.seg_path)
    count_iters = 0
    total_iters = args.n_iters * args.shots
    clip_model.train()


    while count_iters < total_iters:
        for (images, labels) in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            text_features = extract_text_features_train(clip_model, processor, texts, device=device)
            optimizer.zero_grad()
            rois = roi_extractor.extract_from_batch(images)
            # print(rois)
            # image_features = visual_extractor(images, rois, return_attn=True)
            # image_features, attn_weights, roi_indices = visual_extractor(images, rois)
            image_features= visual_extractor(images, rois)
            
            logits_text = clip_model.logit_scale.exp() * image_features @ text_features.T
            logits_visual = clip_model.logit_scale.exp() * image_features @ prototypes().T
            
            loss_text = F.cross_entropy(logits_text, labels)
            loss_visual = F.cross_entropy(logits_visual, labels)

            loss =(1- args.proto_alpha) *  loss_text + args.proto_alpha * loss_visual

            loss.backward()
            # cat prototypes is learnable
            # for name, param in prototypes.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} grad norm: {param.grad.norm().item()}')
            #     else:
            #         print(f'{name} grad is None')
            # for i, param in enumerate(prototypes.parameters()):
            #     print(f"Prototype {i} param norm: {param.norm().item()}")
                
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
    
    print("\nMulticenter Results:", {k: f"{v:.2f}%" for k, v in multicenter_results.items()})
    print("Xiangya Results:", {k: f"{v:.2f}%" for k, v in xiangya_results.items()})
    print("Huaxi Results:", {k: f"{v:.2f}%" for k, v in huaxi_results.items()})


    result_dict = {
        "params": {k: v for k, v in vars(args).items()}, 
        "multicenter_results": {k: f"{v:.2f}%" for k, v in multicenter_results.items()},
        "xiangya_results": {k: f"{v:.2f}%" for k, v in xiangya_results.items()},
        "huaxi_results": {k: f"{v:.2f}%" for k, v in huaxi_results.items()},
    }

    print("[RESULT]", json.dumps(result_dict))
    

class ROIAttentionExtractor(nn.Module):
    def __init__(self, clip_model: nn.Module, query_dim: int = 768, num_heads: int = 4, save_dir: str = './attn_vis'):
        super().__init__()
        self.clip_model = clip_model
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.save_dir = save_dir
        
        assert query_dim % num_heads == 0
        self.head_dim = query_dim // num_heads
        
        self.roi_queries = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.roi_queries)
        
        self.output_proj = nn.Linear(query_dim, query_dim)
        # print(f"output_proj weight shape: {self.output_proj.weight.shape}")
        # print("================================")

    def forward(self, images: torch.Tensor, rois_list: List[Tuple[float, float, float, float]], return_attn=False):
        device = next(self.parameters()).device

        if not isinstance(images, torch.Tensor):
            raise TypeError("images should be Tensor type")
        if images.dim() != 4:
            raise ValueError(f"images should be [B, 3, H, W] dim")

        batch_size = images.size(0)
        if len(rois_list) != batch_size:
            raise ValueError(f"ROI num != images number")

        pixel_values = images.to(device)

        vision_model = self.clip_model.vision_model
        config = vision_model.config
        patch_size = config.patch_size
        image_size = config.image_size
        num_patches_per_side = image_size // patch_size

        outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        roi_features = []

        for i in range(batch_size):
            
            x_min, y_min, x_max, y_max = rois_list[i]
            norm_patch_size = 1.0 / num_patches_per_side
            start_x = int(x_min / norm_patch_size)
            start_y = int(y_min / norm_patch_size)
            end_x = int(x_max / norm_patch_size)
            end_y = int(y_max / norm_patch_size)

            roi_tokens = []
            roi_indices = []
            for y in range(start_y, min(end_y + 1, num_patches_per_side)):
                for x in range(start_x, min(end_x + 1, num_patches_per_side)):
                    patch_x_min = x * norm_patch_size
                    patch_y_min = y * norm_patch_size
                    patch_x_max = (x + 1) * norm_patch_size
                    patch_y_max = (y + 1) * norm_patch_size

                    if (x_min <= patch_x_min and patch_x_max <= x_max and
                        y_min <= patch_y_min and patch_y_max <= y_max):
                        idx = y * num_patches_per_side + x
                        roi_tokens.append(patch_tokens[i, idx])
                        roi_indices.append(idx)


            roi_tokens = torch.stack(roi_tokens, dim=0)  
            
            roi_tokens_multi = roi_tokens.view(-1, self.num_heads, self.head_dim)
            queries = self.roi_queries 
            expanded_queries = queries.repeat(roi_tokens_multi.size(0), 1, 1)
            attn_scores = torch.sum(expanded_queries * roi_tokens_multi, dim=-1) / (self.head_dim ** 0.5)
            attn_scores = attn_scores.transpose(0, 1)  # [num_heads, num_roi_tokens]
            
            attn_weights = F.softmax(attn_scores, dim=-1)  # [num_heads, num_roi_tokens]

            head_outputs = []
            for head_idx in range(self.num_heads):
                head_weights = attn_weights[head_idx]  # [num_roi_tokens]
                head_features = roi_tokens_multi[:, head_idx, :]  # [num_roi_tokens, head_dim]
                head_output = torch.matmul(head_weights, head_features)  # [head_dim]
                head_outputs.append(head_output)
            
            attended_multi = torch.cat(head_outputs, dim=0)  # [query_dim]
            attended = attended_multi.unsqueeze(0)
            attended_proj = self.output_proj(attended)  # [1, query_dim]
            feat = self.clip_model.visual_projection(attended_proj.squeeze(0))
            
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
            roi_features.append(feat)

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
        # cmap = plt.get_cmap("jet")

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

