import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import densenet121
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


import open_clip

class BiomedClipPriorProvider:
    def __init__(self, device='cuda:0'):
        """
        初始化 BiomedCLIP 模型。
        它不需要本地权重路径，会自动从 HuggingFace Hub 下载微软的官方权重。
        """
        self.device = device
        print(f"[*] 正在初始化全模态 BiomedCLIP 先验模块至 {self.device}...")
        
        # 1. 加载 BiomedCLIP 模型与原生的图像预处理管道
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names =[
                    # 1. 终极兜底词汇 (无论什么模态，有病灶就发光)
                    "abnormality", "lesion", "pathological findings", "defect",
                    
                    # 2. 占位性病变 (肿瘤/肿块/囊肿，通杀 CT/MRI/超声)
                    "tumor", "mass", "nodule", "cyst", "neoplasm",
                    
                    # 3. 液体与急性病变 (出血/水肿/积液，通杀 脑部MRI/腹部CT/胸部X光)
                    "hemorrhage", "bleeding", "edema", "effusion", "fluid accumulation",
                    
                    # 4. 炎症与组织破坏 (感染/炎症/骨折/实变)
                    "infection", "inflammation", "fracture", "consolidation", "opacity"
                ]
        
            
            
        print(f"[*] BiomedCLIP 模块加载完毕！")

    def get_heatmaps(self, raw_image, custom_prompts=None,llava_grid_size=24):
        """
        使用 BiomedCLIP 提取图像 Patch 特征，并与文本特征计算余弦相似度，生成热力图。
        保持了与 CheXNet 完全一致的接口。
        """
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')
            
        # 1. 使用 BiomedCLIP 官方的预处理管道 [3, 224, 224]
        image_tensor = self.preprocess(raw_image).unsqueeze(0).to(self.device)
        # prompts = [custom_prompts] if custom_prompts is not None else [f"A medical image with some findings"]
        prompts =[f"A medical image showing {cls}" for cls in self.class_names]
        # prompts.append("A normal medical image without any abnormality") # 追加一个兜底的正常图像提示词
        prompts.append(custom_prompts if custom_prompts is not None else "A medical image with unknown findings")
        with torch.no_grad():
            
            text_tokens = self.tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            self.text_features = F.normalize(text_features, dim=-1) # [Num_classes, 512]
            

            features = self.model.visual.trunk.forward_features(image_tensor)
            
            # b) 剥离 CLS token，只保留 196 个图像块 [1, 196, 768]
            patch_features = features[:, 1:, :] 
            
            patch_features = self.model.visual.head(patch_features) #[1, 196, 512]
                
            # d) 归一化 (计算余弦相似度必备)
            patch_features = F.normalize(patch_features, dim=-1)
            
            # e) 降维去掉 batch，变成 [196, 512]
            patch_features = patch_features.squeeze(0)
            
            
            similarity = self.text_features @ patch_features.T
            
            # 消除负相关（我们只关注强相关的区域，类似于 ReLU）
            cam = F.relu(similarity)
            
            
            grid_size = int(np.sqrt(cam.shape[1])) 
            cam = cam.view(-1, grid_size, grid_size).unsqueeze(0)  #[Num_classes, grid_size, grid_size]

            # 插值放大到 LLaVA 的 24x24 尺寸
            aligned_cam = F.interpolate(
                cam, 
                size=(llava_grid_size, llava_grid_size), 
                mode='bilinear', 
                align_corners=False
            ) #[Num_classes, 24, 24]

        return aligned_cam.squeeze(0) 

    def get_all_topk_indices(self, all_heatmaps, boost_p=0.2):
        """
        你的极速提取代码（完美保留）。
        自适应提取前 20% 最亮区域，并执行边缘斩杀。
        """
        heatmaps = all_heatmaps.clone() 
        
        # 边缘斩杀 (Edge Artifact Masking)
        heatmaps[:, 0:2, :] = 0.0   
        heatmaps[:, -2:, :] = 0.0   
        heatmaps[:, :, 0:2] = 0.0   
        heatmaps[:, :, -2:] = 0.0   

        
        num_classes = heatmaps.shape[0]
        flat_heatmaps = heatmaps.view(num_classes, -1).max(dim=0).values 
        
        # 极速取全局 Top-K
        topk_values, topk_indices = torch.topk(flat_heatmaps, int(boost_p * len(flat_heatmaps)), dim=0)
        
        # 安全过滤 0 值
        valid_mask = topk_values > 1e-5
        valid_indices = topk_indices[valid_mask].tolist()
        
        return valid_indices
    def get_dynamic_threshold_indices(
        self,
        all_heatmaps,
        lambda_thresh=1.0,
        fallback_topk_ratio=0.03,
        eps=1e-5,
    ):
        """
        动态阈值提取显著区域索引。

        阈值公式：
            tau = mean(s_hat) + lambda_thresh * std(s_hat)

        同时保留 TopK fallback，避免热力图分布过平时没有区域被选中。
        """

        """
    动态阈值提取显著区域索引，并返回区域先验 r_i。

    阈值公式：
        tau = mean(s_hat) + lambda_thresh * std(s_hat)

    区域集合：
        A = {i : s_hat_i >= tau} ∪ TopK(s_hat)

    区域先验：
        r_i = I(i in A) * s_hat_i /
              (sum_j I(j in A) * s_hat_j + eps)

    Returns:
        final_indices: List[int]
            被选中的 visual token index。

        ri: Tensor, shape [H * W]
            每个 visual token 对应的 region prior。
            未被选中的位置为 0。
    """

        heatmaps = all_heatmaps.clone()

        # all_heatmaps: [num_classes, H, W]
        num_classes, H, W = heatmaps.shape

        # 边缘斩杀 Edge Artifact Masking
        heatmaps[:, 0:2, :] = 0.0
        heatmaps[:, -2:, :] = 0.0
        heatmaps[:, :, 0:2] = 0.0
        heatmaps[:, :, -2:] = 0.0

        # 多类别热力图融合：每个空间位置取最大响应
        # flat_scores: [H * W]
        flat_scores = heatmaps.reshape(num_classes, -1).max(dim=0).values

        # 安全过滤无效值
        valid_score_mask = flat_scores > eps

        # 如果完全没有有效响应，返回空 index 和全 0 prior
        if valid_score_mask.sum() == 0:
            ri = torch.zeros_like(flat_scores)
            return [], ri

        # 归一化到 [0, 1]，得到 s_hat
        valid_scores = flat_scores[valid_score_mask]
        min_score = valid_scores.min()
        max_score = valid_scores.max()

        s_hat = torch.zeros_like(flat_scores)
        s_hat[valid_score_mask] = (
            valid_scores - min_score
        ) / (max_score - min_score + eps)

        # 只在有效区域上计算 mean 和 std
        valid_s_hat = s_hat[valid_score_mask]
        mean_score = valid_s_hat.mean()
        std_score = valid_s_hat.std(unbiased=False)

        # 动态阈值 tau
        tau = mean_score + lambda_thresh * std_score

        # A_threshold = {i : s_hat_i >= tau}
        threshold_mask = s_hat >= tau

        # TopK fallback：保证至少保留少量 visual tokens
        num_tokens = s_hat.numel()
        topk_num = max(1, int(fallback_topk_ratio * num_tokens))

        topk_values, topk_indices = torch.topk(
            s_hat,
            k=topk_num,
            dim=0
        )

        topk_mask = torch.zeros_like(s_hat, dtype=torch.bool)
        topk_mask[topk_indices[topk_values > eps]] = True

        # A = threshold_indices ∪ TopK
        A_mask = threshold_mask | topk_mask

        # 再过滤一次无响应区域，避免边缘或 0 响应区域进入
        A_mask = A_mask & (s_hat > eps)

        final_indices = torch.nonzero(
            A_mask,
            as_tuple=False
        ).flatten()

        ri = torch.zeros_like(s_hat)

        if final_indices.numel() > 0:
            gamma = 2.0  # >1 放大差异；可试 1.5 / 2.0 / 3.0

            scores = s_hat[final_indices].clamp_min(eps)
            scores = scores.pow(gamma)

            denom = scores.sum() + eps
            ri = scores / denom * final_indices.numel()

        return final_indices, ri
    def __del__(self):
        if hasattr(self, 'model'): del self.model
        try: torch.cuda.empty_cache()
        except: pass



def generate_topk_heatmap(
    all_heatmaps,
    boost_p=0.2,
    edge_kill=2,
    eps=1e-5,
    normalize=True,
    show_topk=True,
    save_path=None,
    title="Aggregated Saliency Heatmap"
):
    """
    根据 get_all_topk_indices 的逻辑生成配套热力图。

    Args:
        all_heatmaps: torch.Tensor, shape = [num_classes, H, W]
            多类别 saliency heatmaps。
        boost_p: float
            选取前 boost_p 比例的高响应区域，例如 0.2 表示 Top 20%。
        edge_kill: int
            边缘斩杀宽度，默认和你的代码一致为 2。
        eps: float
            过滤近似 0 值的阈值。
        normalize: bool
            是否将热力图归一化到 [0, 1]，便于可视化。
            如果想完全保留原始数值色阶，可以设为 False。
        show_topk: bool
            是否在热力图上标出 Top-K 区域。
        save_path: str or None
            如果不为 None，则保存图片。
        title: str
            图标题。

    Returns:
        heatmap_np: np.ndarray, shape = [H, W]
            聚合后的热力图。
        topk_mask_np: np.ndarray, shape = [H, W]
            Top-K 区域 mask。
        valid_indices: list[int]
            有效 Top-K flatten indices。
    """

    assert isinstance(all_heatmaps, torch.Tensor), "all_heatmaps must be a torch.Tensor"
    assert all_heatmaps.ndim == 3, "all_heatmaps shape should be [num_classes, H, W]"

    heatmaps = all_heatmaps.detach().clone().float()

    num_classes, H, W = heatmaps.shape

    # --------------------------------------------------
    # 1. 边缘斩杀，与原代码保持一致
    # --------------------------------------------------
    if edge_kill > 0:
        heatmaps[:, :edge_kill, :] = 0.0
        heatmaps[:, -edge_kill:, :] = 0.0
        heatmaps[:, :, :edge_kill] = 0.0
        heatmaps[:, :, -edge_kill:] = 0.0

    # --------------------------------------------------
    # 2. 多类别 heatmap 聚合
    #    对每个 patch 取所有类别中的最大响应
    # --------------------------------------------------
    flat_heatmaps = heatmaps.view(num_classes, -1).max(dim=0).values

    total_patches = flat_heatmaps.numel()
    k = max(1, int(boost_p * total_patches))

    # --------------------------------------------------
    # 3. Top-K 提取
    # --------------------------------------------------
    topk_values, topk_indices = torch.topk(flat_heatmaps, k, dim=0)

    valid_mask = topk_values > eps
    valid_indices = topk_indices[valid_mask]

    # --------------------------------------------------
    # 4. 恢复成二维 heatmap
    # --------------------------------------------------
    heatmap_2d = flat_heatmaps.view(H, W)

    topk_mask = torch.zeros_like(flat_heatmaps, dtype=torch.bool)
    topk_mask[valid_indices] = True
    topk_mask_2d = topk_mask.view(H, W)

    # --------------------------------------------------
    # 5. 是否归一化
    # --------------------------------------------------
    if normalize:
        min_val = heatmap_2d.min()
        max_val = heatmap_2d.max()

        if max_val > min_val:
            heatmap_vis = (heatmap_2d - min_val) / (max_val - min_val)
        else:
            heatmap_vis = torch.zeros_like(heatmap_2d)
    else:
        heatmap_vis = heatmap_2d

    heatmap_np = heatmap_vis.cpu().numpy()
    raw_heatmap_np = heatmap_2d.cpu().numpy()
    topk_mask_np = topk_mask_2d.cpu().numpy()

    # --------------------------------------------------
    # 6. 可视化
    # --------------------------------------------------
    plt.figure(figsize=(6, 5))

    im = plt.imshow(heatmap_np, cmap="jet")

    if normalize:
        cbar = plt.colorbar(im)
        cbar.set_label("Normalized Saliency")
    else:
        cbar = plt.colorbar(im)
        cbar.set_label("Raw Saliency Value")

    if show_topk and topk_mask_np.any():
        # 用等高线描出 Top-K 区域
        plt.contour(
            topk_mask_np.astype(np.float32),
            levels=[0.5],
            colors="white",
            linewidths=1.5
        )

    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return raw_heatmap_np, topk_mask_np, valid_indices.cpu().tolist()


CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class CheXNetPriorProvider:
    def __init__(self, weight_path='chexnet_weights.pth', device='cuda:0'):
        """
        初始化：加载模型并提取出最后一层线性分类器的权重。
        """
        self.device = device
        print(f"[*] 正在初始化极速版 CheXNet 先验模块至 {self.device}...")
        
        # 1. 定义网络
        self.model = densenet121(pretrained=False)
        num_ftrs = self.model.classifier.in_features  # DenseNet121 是 1024 维特征
        
        # 保持与 CheXNet 结构一致
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 14),
            torch.nn.Sigmoid()
        )
        
        # 2. 加载权重
        checkpoint = torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval() # 开启评估模式
        
        # ==========================================================
        # 🌟 核心工程技巧：把分类器的权重单独“抠”出来缓存！
        # 这个权重矩阵形状是 [14, 1024]，代表 14 种疾病对 1024 个特征通道的关注度
        # ==========================================================
        self.fc_weights = self.model.classifier[0].weight.data.clone().to(self.device)
        
        # =========================================================
        # 🌟 新增：专门为 CheXNet 准备的图像预处理管道
        # =========================================================
        self.transform = T.Compose([
            T.Resize((224, 224), antialias=True), # 强行缩放到 CheXNet 需要的尺寸
            T.ToTensor(),                         # 转为 Tensor，并将像素值缩放至[0, 1]
            T.Normalize(                          # ImageNet 标准归一化（CheXNet 训练必须项）
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        print("[*] 极速版模块加载完毕！")

    def get_heatmaps(self, raw_image, llava_grid_size=24):
        """
        纯前向传播，时间复杂度 O(1)，一次性返回 14 张热力图！
        输入: input_tensor[1, 3, 224, 224] (你的X光片)
        输出: tensor 形状为 [14, 24, 24]，包含了 14 种病灶的空间热力图
        """
        # 🌟 关键防御：医学 X 光片通常是单通道灰度图 ('L' 模式)，
        # 而预处理管道需要 3 通道 ('RGB' 模式)，必须先转换！
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')
        
        # 1. 直接对最原始的 PIL 图像应用标准转换
        clean_tensor = self.transform(raw_image) # 变成形状 [3, 224, 224]

         # 2. 增加 Batch 维度并送入显卡
        clean_tensor = clean_tensor.unsqueeze(0).to(self.device) # 变成[1, 3, 224, 224]
        
        with torch.no_grad(): # 绝对不计算梯度，节省大量显存和时间！
            # 1. 执行一次前向传播，只运行卷积层，拿到空间特征图
            # features 形状:[1, 1024, 7, 7] (7x7是 DenseNet 输出的空间分辨率)
            features = self.model.features(clean_tensor)
            features = F.relu(features, inplace=True)
            
        # 2. 🌟 矩阵乘法魔法 (Einsum) 🌟
        # 将分类权重 [14, 1024] 与特征图[1, 1024, 7, 7] 相乘
        # 这一步数学含义：直接算出 14 种疾病在 7x7 网格上的激活程度！
        cam = torch.einsum('nc,bchw->bnhw', self.fc_weights, features) # 输出[1, 14, 7, 7]
        
        # 消除负数激活（只关注有正向促进作用的特征）
        cam = F.relu(cam)
        
        # 3. 将 7x7 的热力图插值放大到 LLaVA 的 24x24 尺寸
        aligned_cam = F.interpolate(
            cam, 
            size=(llava_grid_size, llava_grid_size), 
            mode='bilinear', 
            align_corners=False
        ) # 输出[1, 14, 24, 24]

        # 去掉 batch 维度，返回[14, 24, 24] 的张量
        return aligned_cam.squeeze(0)

    def get_all_topk_indices(self, all_heatmaps, boost_p=0.3):
        """
        一次性返回 14 种疾病的 Top-K 索引，包含边缘斩杀过滤。
        输入: all_heatmaps [14, 24, 24]
        输出: List[List[int]]，长度为 14，每个子列表包含 boost_num 个索引
        """
        # clone 一份，防止修改原始的热力图张量
        heatmaps = all_heatmaps.clone() 
        
        # =========================================================
        # 🛡️ 核心修复：无情斩杀 14 张图的边缘伪影！
        # 强行把最上面 2 行、最下面 2 行、最左侧 2 列、最右侧 2 列归零！
        # 彻底干掉黑框、PORTABLE 字母和 CNN 零填充白边的干扰
        # =========================================================
        heatmaps[:, 0:2, :] = 0.0   # 顶端 2 行归零
        heatmaps[:, -2:, :] = 0.0   # 底端 2 行归零
        heatmaps[:, :, 0:2] = 0.0   # 左侧 2 列归零
        heatmaps[:, :, -2:] = 0.0   # 右侧 2 列归零

        # 展平为 [14, 576] 维张量
        flat_heatmaps = heatmaps.view(14, -1).max(dim=0).values 
        
        # 极速取 Top-K (沿着维度 1，也就是在这 576 个坐标里挑)
        topk_values, topk_indices = torch.topk(flat_heatmaps, int(boost_p*len(flat_heatmaps)), dim=0)
        
        valid_mask = topk_values > 1e-5
        
        # 使用掩码筛选出有效的索引，并转换为普通的 Python 列表
        valid_indices = topk_indices[valid_mask].tolist()
        
        return valid_indices
    def __del__(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'fc_weights'): del self.fc_weights
        try: torch.cuda.empty_cache()
        except: pass



def visualize_and_save_single_topk_heatmap(
    raw_image: Image.Image,
    all_heatmaps: torch.Tensor,
    boost_p: float = 0.2,
    edge_kill: int = 2,
    eps: float = 1e-5,
    alpha: float = 0.4,
    draw_topk_contour: bool = True,
    use_topk_value_map: bool = True,
    save_path: str = "single_topk_heatmap.png"
):
    """
    生成 1 张与 get_all_topk_indices 逻辑一致的融合热力图，并保存。

    核心逻辑：
        1. 对 all_heatmaps 做边缘斩杀
        2. 对所有类别 heatmap 做 max 聚合
        3. 全局 Top-K
        4. 过滤 0 值
        5. 使用 topk_values 构造 Top-K 数值热力图

    参数:
        raw_image:
            原始图像，PIL.Image。

        all_heatmaps:
            Tensor，形状 [num_classes, H, W]，例如 [14, 24, 24]。

        boost_p:
            取前 boost_p 比例的高响应区域。

        edge_kill:
            边缘斩杀宽度，默认 2，与你原代码一致。

        eps:
            有效响应阈值，默认 1e-5。

        alpha:
            热力图叠加透明度。

        draw_topk_contour:
            是否绘制 Top-K 区域轮廓。

        use_topk_value_map:
            True:
                绘制只有 Top-K 区域有响应的热力图。
                颜色强度来自 topk_values。
            False:
                绘制完整 aggregated_heatmap。
                但是仍然会用 Top-K mask 画轮廓。

        save_path:
            保存路径。

    返回:
        result: dict
            {
                "valid_indices": list[int],
                "valid_values": np.ndarray,
                "aggregated_heatmap": np.ndarray,
                "topk_value_map": np.ndarray,
                "topk_mask": np.ndarray
            }
    """

    print("[*] 正在生成单张 Top-K 数值热力图可视化...")

    # --------------------------------------------------
    # 1. 准备原图
    # --------------------------------------------------
    if raw_image.mode != "RGB":
        raw_image = raw_image.convert("RGB")

    img_arr = np.array(raw_image)
    img_height, img_width = img_arr.shape[:2]
    img_arr_float = img_arr.astype(np.float32) / 255.0

    # --------------------------------------------------
    # 2. 准备 heatmaps，并执行边缘斩杀
    # --------------------------------------------------
    assert isinstance(all_heatmaps, torch.Tensor), "all_heatmaps 必须是 torch.Tensor"
    assert all_heatmaps.ndim == 3, "all_heatmaps 应为 [num_classes, H, W]"

    heatmaps = all_heatmaps.detach().clone().float()
    num_classes, h, w = heatmaps.shape

    if edge_kill > 0:
        heatmaps[:, 0:edge_kill, :] = 0.0
        heatmaps[:, -edge_kill:, :] = 0.0
        heatmaps[:, :, 0:edge_kill] = 0.0
        heatmaps[:, :, -edge_kill:] = 0.0

    # --------------------------------------------------
    # 3. 与 get_all_topk_indices 一致：
    #    所有类别在每个 patch 上取最大响应
    # --------------------------------------------------
    flat_heatmaps = heatmaps.view(num_classes, -1).max(dim=0).values
    aggregated_heatmap = flat_heatmaps.view(h, w)

    # --------------------------------------------------
    # 4. 全局 Top-K
    # --------------------------------------------------
    total_num = flat_heatmaps.numel()
    k = max(1, int(boost_p * total_num))

    topk_values, topk_indices = torch.topk(flat_heatmaps, k, dim=0)

    # --------------------------------------------------
    # 5. 安全过滤 0 值
    # --------------------------------------------------
    valid_mask = topk_values > eps

    valid_indices = topk_indices[valid_mask]
    valid_values = topk_values[valid_mask]

    # --------------------------------------------------
    # 6. 构造 Top-K mask
    # --------------------------------------------------
    topk_mask_flat = torch.zeros_like(flat_heatmaps, dtype=torch.bool)
    topk_mask_flat[valid_indices] = True
    topk_mask = topk_mask_flat.view(h, w)

    # --------------------------------------------------
    # 7. 关键修正：
    #    使用 topk_values 构造 Top-K 数值热力图
    # --------------------------------------------------
    topk_value_map_flat = torch.zeros_like(flat_heatmaps)
    topk_value_map_flat[valid_indices] = valid_values
    topk_value_map = topk_value_map_flat.view(h, w)

    # --------------------------------------------------
    # 8. 选择用于可视化的 heatmap
    # --------------------------------------------------
    if use_topk_value_map:
        heatmap_for_vis = topk_value_map
        vis_title = "Top-K Value Heatmap"
    else:
        heatmap_for_vis = aggregated_heatmap
        vis_title = "Aggregated Saliency Heatmap"

    heatmap_np = heatmap_for_vis.cpu().numpy()

    # --------------------------------------------------
    # 9. 归一化仅用于显示
    #    注意：Top-K 提取和 topk_value_map 构造均基于原始数值
    # --------------------------------------------------
    min_val = heatmap_np.min()
    max_val = heatmap_np.max()

    if max_val > min_val:
        heatmap_normalized = (heatmap_np - min_val) / (max_val - min_val)
    else:
        heatmap_normalized = np.zeros_like(heatmap_np, dtype=np.float32)

    # --------------------------------------------------
    # 10. Resize 到原图尺寸
    # --------------------------------------------------
    heatmap_pil = Image.fromarray((heatmap_normalized * 255).astype(np.uint8))

    try:
        bicubic_mode = Image.Resampling.BICUBIC
        nearest_mode = Image.Resampling.NEAREST
    except AttributeError:
        bicubic_mode = Image.BICUBIC
        nearest_mode = Image.NEAREST

    heatmap_resized_pil = heatmap_pil.resize(
        (img_width, img_height),
        resample=bicubic_mode
    )
    heatmap_resized = np.array(heatmap_resized_pil).astype(np.float32) / 255.0

    # Resize Top-K mask，用于画轮廓
    topk_mask_np = topk_mask.cpu().numpy().astype(np.uint8) * 255
    topk_mask_pil = Image.fromarray(topk_mask_np)
    topk_mask_resized_pil = topk_mask_pil.resize(
        (img_width, img_height),
        resample=nearest_mode
    )
    topk_mask_resized = (np.array(topk_mask_resized_pil) > 127).astype(np.uint8)

    # --------------------------------------------------
    # 11. JET 伪彩色并融合
    # --------------------------------------------------
    jet_cmap = plt.get_cmap("jet")
    colormap_rgba = jet_cmap(heatmap_resized)
    colormap_float = colormap_rgba[..., :3].astype(np.float32)

    blended = (1.0 - alpha) * img_arr_float + alpha * colormap_float
    blended = np.clip(blended, 0, 1)

    # --------------------------------------------------
    # 12. 绘图
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image", fontweight="bold", fontsize=14)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_np, cmap="jet")
    axes[1].set_title(vis_title + " (Raw Values)", fontweight="bold", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(blended)

    if draw_topk_contour and topk_mask_resized.sum() > 0:
        axes[2].contour(
            topk_mask_resized.astype(np.float32),
            levels=[0.5],
            colors="white",
            linewidths=1.2
        )

    axes[2].set_title("Overlay on Image", fontweight="bold", fontsize=14)
    axes[2].axis("off")

    # --------------------------------------------------
    # 13. 保存
    # --------------------------------------------------
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[*] ✅ 单张热力图已保存至: {save_path}")
    print(f"[*] Top-K 有效 patch 数量: {len(valid_indices)} / {k}")
    print(f"[*] Top-K value range: [{valid_values.min().item() if len(valid_values) > 0 else 0:.6f}, "
          f"{valid_values.max().item() if len(valid_values) > 0 else 0:.6f}]")

    return {
        "valid_indices": valid_indices.cpu().tolist(),
        "valid_values": valid_values.cpu().numpy(),
        "aggregated_heatmap": aggregated_heatmap.cpu().numpy(),
        "topk_value_map": topk_value_map.cpu().numpy(),
        "topk_mask": topk_mask.cpu().numpy()
    }
