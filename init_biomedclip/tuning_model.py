import torch
from utils.prompt import descriptions
import json
from peft_model import apply_peft_to_model
import torch.nn as nn
from typing import List, Tuple, Optional
import torch.nn.functional as F
from get_roi import ROIExtractor
from utils.utils import evaluate_lora

def extract_text_features(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)

    # unwrap 一次
    m = model.base_model.model if hasattr(model, "base_model") else model

    text_features = m.encode_text(tokens)
    return text_features / text_features.norm(dim=-1, keepdim=True)


def run_lora(args, clip_model, processor,tokenizer, dataset, train_loader, val_loader, test_loader,xiangya_test_loader,huaxi_test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    texts = [descriptions[classname] for classname in dataset.classnames]
    print(texts)
    print("run lora")
    print("Applying PEFT to model...")

    base_clip_model = clip_model
    vision_lora_config = json.loads(args.vision_lora_config)
    text_lora_config = json.loads(args.text_lora_config)

    clip_model = apply_peft_to_model(
        base_clip_model,
        peft_type = args.peft_type,
        use_vision_peft =args.use_vision_peft,
        use_text_peft =args.use_text_peft,
        vision_config_kwargs=vision_lora_config,
        text_config_kwargs=text_lora_config
    ).to(device)

    visual_extractor = ROIAttentionExtractor(clip_model, num_heads=4).to(device)

    num_classes = len(dataset.classnames)

    feat_dim = get_visual_feat_dim(clip_model)
    prototypes = VisualPrototypes(num_classes, feat_dim).to(device)
    print(prototypes().shape) # [5 * 512]

    params = []
    params += [p for p in clip_model.parameters() if p.requires_grad]
    params += [visual_extractor.roi_queries]
    params += list(visual_extractor.output_proj.parameters())
    params += list(prototypes.parameters())
    assert len({id(p) for p in params}) == len(params), "optimizer params 有重复（同一个参数被加了多次）"


    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_iters * args.shots, eta_min=1e-6)
    print("proto is learnable?")
    print([p.requires_grad for p in prototypes.parameters()])
    roi_extractor = ROIExtractor(model_path= args.seg_path)
    count_iters = 0
    total_iters = args.n_iters * args.shots
    clip_model.train()
    visual_extractor.train()
    prototypes.train()


    text_encoder_trainable = args.use_text_peft
    if not text_encoder_trainable:
        with torch.no_grad():
            text_features = extract_text_features(clip_model, tokenizer, texts, device=device)
        print("[INFO] 文本编码器冻结，text_features 已预计算")
    else:
        print("[INFO] 文本编码器可训练，text_features 将在每次迭代时重新计算")
        text_features = None  # 将在循环中计算

    while count_iters < total_iters:
        for (images, labels) in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 如果文本编码器可训练，每次迭代重新计算 text_features
            if text_encoder_trainable:
                text_features = extract_text_features(clip_model, tokenizer, texts, device=device)
            optimizer.zero_grad()
            rois = roi_extractor.extract_from_batch(images)
            image_features= visual_extractor(images, rois)

            scale = get_logit_scale(clip_model)
            logits_text = scale * (image_features @ text_features.T)
            logits_visual = scale * (image_features @ prototypes().T)

            loss_text = F.cross_entropy(logits_text, labels)
            loss_visual = F.cross_entropy(logits_visual, labels)
            loss =(1- args.proto_alpha) *  loss_text + args.proto_alpha * loss_visual
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            count_iters += 1
            if count_iters >= total_iters:
                break

        print_interval = 5
        if count_iters % print_interval == 0 or count_iters >= total_iters:
            val_results = evaluate_lora(
                args, clip_model, val_loader,
                text_features, visual_extractor, prototypes, roi_extractor,
                tokenizer=tokenizer if args.use_text_peft else None,
                texts=texts if args.use_text_peft else None
            )
            acc_val = val_results.get("acc_5class", 0.0)
            # print(f"[Iter {count_iters}/{total_iters}] Loss: {loss.item():.4f} | Val Acc: {acc_val:.2f}%")
            print(f"[Iter {count_iters}/{total_iters}] "
                f"Total Loss: {loss.item():.4f} | "
                f"Text Loss: {loss_text.item():.4f} | "
                f"Proto Loss: {loss_visual.item():.4f} | "
                f"Val Acc: {acc_val:.2f}%")


    multicenter_results = evaluate_lora(args, clip_model, test_loader, text_features, visual_extractor, prototypes, roi_extractor,
                                        tokenizer=tokenizer if args.use_text_peft else None,
                                        texts=texts if args.use_text_peft else None)
    xiangya_results = evaluate_lora(args, clip_model, xiangya_test_loader, text_features, visual_extractor, prototypes, roi_extractor,
                                     tokenizer=tokenizer if args.use_text_peft else None,
                                     texts=texts if args.use_text_peft else None)
    huaxi_results = evaluate_lora(args, clip_model, huaxi_test_loader, text_features, visual_extractor, prototypes, roi_extractor,
                                   tokenizer=tokenizer if args.use_text_peft else None,
                                   texts=texts if args.use_text_peft else None)
    
    # 统一打印格式（字典方式）
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



def get_logit_scale(model):
    m = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    if hasattr(m, "logit_scale"):
        return m.logit_scale.exp()
    # 如果模型没有 logit_scale 属性，返回默认值 100.0（CLIP 模型的常见值）
    device = next(model.parameters()).device if hasattr(model, 'parameters') and next(model.parameters(), None) is not None else torch.device('cpu')
    return torch.tensor(100.0, device=device)


def get_visual_feat_dim(clip_model):
    m = clip_model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m.visual.head.proj.out_features



class VisualPrototypes(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        init_proto = torch.empty(num_classes, feat_dim)
        nn.init.orthogonal_(init_proto)  
        self.prototypes = nn.Parameter(F.normalize(init_proto, p=2, dim=1))

    def forward(self):
        return F.normalize(self.prototypes, dim=1)

def _build_roi_keep_mask(
    rois_list: List[Tuple[float, float, float, float]],
    grid: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    从 rois_list 构建 [B, seq_len] 的 bool mask，seq_len = 1+P（CLS + patch）。
    CLS 位置为 True；patch 位置：ROI 内为 True，非 ROI 为 False。
    严格：patch 完全落在 ROI 内才算 ROI。
    """
    P = grid * grid
    seq_len = 1 + P
    norm_patch_size = 1.0 / grid
    roi_keep_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    roi_keep_mask[:, 0] = True  # CLS 恒为 True
    for i in range(batch_size):
        x_min, y_min, x_max, y_max = rois_list[i]
        start_x = int(x_min / norm_patch_size)
        start_y = int(y_min / norm_patch_size)
        end_x = int(x_max / norm_patch_size)
        end_y = int(y_max / norm_patch_size)
        for yy in range(start_y, min(end_y + 1, grid)):
            for xx in range(start_x, min(end_x + 1, grid)):
                patch_x_min = xx * norm_patch_size
                patch_y_min = yy * norm_patch_size
                patch_x_max = (xx + 1) * norm_patch_size
                patch_y_max = (yy + 1) * norm_patch_size
                if (x_min <= patch_x_min and patch_x_max <= x_max and
                    y_min <= patch_y_min and patch_y_max <= y_max):
                    idx = yy * grid + xx
                    roi_keep_mask[i, 1 + idx] = True
    return roi_keep_mask


def _vit_forward_with_roi_masking(
    vit: nn.Module,
    pixel_values: torch.Tensor,
    roi_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """
    逐 block 跑 ViT，每个 block 输出后对 hidden_states 做：
    hidden_states = hidden_states * roi_keep_mask.unsqueeze(-1).float()
    非 ROI 的 patch token 直接置零；ROI 与 CLS 保持不变。
    返回 [B, 1+P, C]。roi_keep_mask: [B, seq_len]，seq_len=1+P。
    """
    B = pixel_values.shape[0]
    x = vit.patch_embed(pixel_values)  # [B, P, C]
    P, C = x.shape[1], x.shape[2]
    cls_token = getattr(vit, "cls_token", None)
    pos_embed = getattr(vit, "pos_embed", None)
    if cls_token is not None:
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+P, C]
    if pos_embed is not None:
        if pos_embed.shape[1] != x.shape[1]:
            pos_embed = pos_embed[:, : x.shape[1]]
        x = x + pos_embed
    pos_drop = getattr(vit, "pos_drop", None)
    if pos_drop is not None:
        x = pos_drop(x)
    norm_pre = getattr(vit, "norm_pre", None)
    if norm_pre is not None:
        x = norm_pre(x)
    mask = roi_keep_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
    for block in vit.blocks:
        x = block(x)  # [B, 1+P, C]
        x = x * mask
    return x


class ROIAttentionExtractor(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        query_dim: int = 768,
        num_heads: int = 4,
        use_intermediate_roi_masking: bool = True,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.use_intermediate_roi_masking = use_intermediate_roi_masking

        assert query_dim % num_heads == 0, "query_dim必须能被num_heads整除"
        self.head_dim = query_dim // num_heads

        # [1, H, D]
        self.roi_queries = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.roi_queries)

        self.output_proj = nn.Linear(query_dim, query_dim)

    # -------- 关键：自动兼容 PEFT 的包装层 --------
    def _unwrap_clip(self) -> nn.Module:
        """
        如果 clip_model 是 peft 的 PeftModel，它通常把原模型包在 base_model.model 里。
        这个函数保证我们拿到真正的 open_clip CustomTextCLIP 实例。
        """
        m = self.clip_model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            return m.base_model.model
        return m

    def forward(
        self,
        images: torch.Tensor,
        rois_list: List[Tuple[float, float, float, float]],
        return_attn: bool = False,
    ):
        device = next(self.parameters()).device

        if not isinstance(images, torch.Tensor):
            raise TypeError("images 应该是 Tensor 格式，请确认图像是否经过 transform。")
        if images.dim() != 4:
            raise ValueError(f"images 应该是 [B, 3, H, W] 维度，但现在是 {images.shape}")

        batch_size = images.size(0)
        if len(rois_list) != batch_size:
            raise ValueError(f"ROI数量({len(rois_list)})与图像数量({batch_size})不匹配")

        pixel_values = images.to(device)

        # === 1) 取到真正的 open_clip 模型结构（兼容 PEFT） ===
        base_clip = self._unwrap_clip()

        # open_clip: visual.trunk 是 timm ViT
        vit = base_clip.visual.trunk
        proj_768_to_512 = base_clip.visual.head.proj  # Linear(768->512)

        # 推断 P、grid（用于构建 roi_keep_mask 与后续 ROI 选择）
        pos_embed = getattr(vit, "pos_embed", None)
        if pos_embed is not None:
            P = pos_embed.shape[1] - 1  # 1+patch
        else:
            P = getattr(vit.patch_embed, "num_patches", 196)
        if isinstance(P, torch.Tensor):
            P = int(P.item())
        grid = int(P ** 0.5)
        if grid * grid != P:
            raise ValueError(f"num_patches={P} 不是平方数，无法推断 grid，ROI 映射会不正确。")
        norm_patch_size = 1.0 / grid

        if self.use_intermediate_roi_masking:
            roi_keep_mask = _build_roi_keep_mask(rois_list, grid, batch_size, device)
            x = _vit_forward_with_roi_masking(vit, pixel_values, roi_keep_mask)  # [B, 1+P, C]
        else:
            tokens_out = {}
            def hook_fn(module, inp, out):
                tokens_out["x"] = out
            h = vit.blocks[-1].register_forward_hook(hook_fn)
            _ = vit(pixel_values)
            h.remove()
            if "x" not in tokens_out:
                raise RuntimeError("无法从 vit 的 hook 捕获 token 输出，请检查 timm/open_clip 版本的 forward 返回。")
            x = tokens_out["x"]

        if x.dim() != 3 or x.size(-1) != self.query_dim:
            raise RuntimeError(f"捕获到的 token shape 异常: {tuple(x.shape)}，期望 [..., {self.query_dim}]")

        patch_tokens = x[:, 1:, :]  # [B, P, 768]
        B, P, C = patch_tokens.shape

        roi_features = []
        attn_debug = [] if return_attn else None

        # === 4) ROI -> 严格选 patch（完全落在 ROI 内）===
        for i in range(batch_size):
            x_min, y_min, x_max, y_max = rois_list[i]

            # 计算 ROI 覆盖到的 patch index 范围（你原逻辑保留）
            start_x = int(x_min / norm_patch_size)
            start_y = int(y_min / norm_patch_size)
            end_x = int(x_max / norm_patch_size)
            end_y = int(y_max / norm_patch_size)

            roi_tokens = []
            roi_indices = []

            for yy in range(start_y, min(end_y + 1, grid)):
                for xx in range(start_x, min(end_x + 1, grid)):
                    patch_x_min = xx * norm_patch_size
                    patch_y_min = yy * norm_patch_size
                    patch_x_max = (xx + 1) * norm_patch_size
                    patch_y_max = (yy + 1) * norm_patch_size

                    # 严格：patch 完全落在 ROI 内
                    if (x_min <= patch_x_min and patch_x_max <= x_max and
                        y_min <= patch_y_min and patch_y_max <= y_max):
                        idx = yy * grid + xx
                        roi_tokens.append(patch_tokens[i, idx])
                        roi_indices.append(idx)

            if not roi_tokens:
                raise ValueError(f"第{i}张图像的ROI {rois_list[i]}未包含任何有效patch（严格模式导致为空）")

            roi_tokens = torch.stack(roi_tokens, dim=0)  # [N, 768]
            N = roi_tokens.size(0)

            roi_tokens_multi = roi_tokens.view(N, self.num_heads, self.head_dim)  # [N, H, D]

            queries = self.roi_queries                                  # [1, H, D]
            expanded_queries = queries.expand(N, -1, -1)                 # [N, H, D]

            attn_scores = torch.sum(expanded_queries * roi_tokens_multi, dim=-1) / (self.head_dim ** 0.5)  # [N, H]
            attn_scores = attn_scores.transpose(0, 1)                    # [H, N]
            attn_weights = F.softmax(attn_scores, dim=-1)                # [H, N]

            head_outputs = []
            for head_idx in range(self.num_heads):
                head_weights = attn_weights[head_idx]                   # [N]
                head_features = roi_tokens_multi[:, head_idx, :]        # [N, D]
                head_output = torch.matmul(head_weights, head_features) # [D]
                head_outputs.append(head_output)

            attended_multi = torch.cat(head_outputs, dim=0)              # [768]
            attended = attended_multi.unsqueeze(0)                       # [1, 768]
            attended_proj = self.output_proj(attended).squeeze(0)        # [768]

            feat = proj_768_to_512(attended_proj)                        # [512]
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)

            roi_features.append(feat)

            if return_attn:
                attn_debug.append({
                    "roi_indices": roi_indices,
                    "attn_weights": attn_weights.detach().cpu(),  # [H, N]
                })

        roi_features = torch.stack(roi_features, dim=0)  # [B, 512]

        if return_attn:
            return roi_features, attn_debug
        return roi_features
