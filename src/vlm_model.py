"""Kiến trúc mô hình Vision-Language Model hoàn chỉnh (RealisticMiniVLM).

Gồm 3 khối kiến trúc nền tảng:
1. Vision Encoder: Trích xuất đặc trưng không gian từ các Image Patches.
2. Modality Projector: Chiếu biểu diễn thị giác vào không gian ngữ nghĩa của LLM.
3. LLM Backbone: Ngăn xếp Transformer Decoder xử lý chuỗi đa phương thức với Multimodal Causal Mask.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vlm_utils import VOCAB, ID2WORD


class VisionEncoder(nn.Module):
    """Bộ mã hóa thị giác trích xuất 16 visual tokens từ ảnh 64x64."""
    def __init__(self, in_channels: int = 3, d_v: int = 64, patch_size: int = 16, num_patches: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_v = d_v
        
        # Chiếu tích chập tương đương Patch Embedding (P x P)
        self.patch_conv = nn.Conv2d(in_channels, d_v, kernel_size=patch_size, stride=patch_size)
        # Vector nhúng vị trí không gian 2D học được
        self.vis_pos = nn.Parameter(torch.randn(1, num_patches, d_v) * 0.02)
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: [B, C, H, W] -> [B, d_v, H/P, W/P] -> [B, num_patches, d_v]
        patches = self.patch_conv(img).flatten(2).transpose(1, 2)
        return patches + self.vis_pos


class MLPProjector(nn.Module):
    """Khối chiếu đa phương thức MLP 2 tầng với hàm kích hoạt GELU."""
    def __init__(self, d_v: int = 64, d_llm: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_v, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VLMDecoderLayer(nn.Module):
    """Tầng Transformer Decoder hỗ trợ trích xuất ma trận Attention xuyên phương thức."""
    def __init__(self, d_model: int = 64, nhead: int = 2, dim_feedforward: int = 128):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN Self-Attention trên toàn bộ chuỗi đa phương thức
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            norm_x, norm_x, norm_x,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=True
        )
        x = x + attn_out
        
        # Feed-Forward Network với kết nối tắt Residual
        ffn_out = self.linear2(self.activation(self.linear1(self.norm2(x))))
        x = x + ffn_out
        return x, attn_weights


class RealisticMiniVLM(nn.Module):
    """Mô hình Vision-Language Model hoàn chỉnh tích hợp 3 trụ cột kỹ thuật:
    1. Vision Encoder (ViT patch embedding)
    2. Modality Projector (MLP chiếu đặc trưng)
    3. LLM Backbone (Ngăn xếp Decoder hỗ trợ Multimodal Causal Masking)
    """
    def __init__(self, vocab_size: int = len(VOCAB), d_v: int = 64, d_llm: int = 64, max_text_len: int = 32):
        super().__init__()
        self.d_v = d_v
        self.d_llm = d_llm
        self.vocab_size = vocab_size
        
        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(in_channels=3, d_v=d_v, patch_size=16, num_patches=16)
        
        # 2. Modality Projector
        self.projector = MLPProjector(d_v=d_v, d_llm=d_llm)
        
        # 3. Text & Positional Embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_llm)
        self.text_pos = nn.Embedding(max_text_len, d_llm)
        
        # 4. LLM Backbone (2 tầng Decoder)
        self.layer1 = VLMDecoderLayer(d_llm, nhead=2)
        self.layer2 = VLMDecoderLayer(d_llm, nhead=2)
        
        # 5. Language Model Head
        self.lm_head = nn.Linear(d_llm, vocab_size)
        
    def forward(self, img: torch.Tensor, text_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Thực hiện Forward Pass đa phương thức.
        
        Args:
            img: Tensor ảnh [B, 3, H, W]
            text_ids: Tensor token IDs văn bản [B, N_t]
            
        Returns:
            logits: Phân phối điểm số từ vựng [B, 16 + N_t, vocab_size]
            attn_weights: Ma trận attention từ tầng cuối cùng [B, seq_len, seq_len]
            n_vis: Số lượng visual tokens (16)
            n_text: Số lượng text tokens (N_t)
        """
        B = img.shape[0]
        
        # BƯỚC 1: Xử lý hình ảnh -> 16 visual tokens
        vis_features = self.vision_encoder(img)              # [B, 16, d_v]
        vis_tokens = self.projector(vis_features)            # [B, 16, d_llm]
        n_vis = vis_tokens.shape[1]
        
        # BƯỚC 2: Xử lý văn bản -> N_t text tokens
        n_text = text_ids.shape[1]
        t_pos = torch.arange(n_text, device=text_ids.device).unsqueeze(0)
        text_tokens = self.tok_embed(text_ids) + self.text_pos(t_pos)  # [B, N_t, d_llm]
        
        # BƯỚC 3: Ghép nối đa phương thức (Multimodal Concatenation)
        multimodal_seq = torch.cat([vis_tokens, text_tokens], dim=1)  # [B, n_vis + n_text, d_llm]
        seq_len = multimodal_seq.shape[1]
        
        # BƯỚC 4: Tạo mặt nạ Attention Đa phương thức (Multimodal Causal Mask)
        # Visual tokens (0..15): 2 chiều (Bidirectional)
        # Text tokens (16..end): Nhìn thấy toàn bộ ảnh + các từ phía trước (Causal)
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=text_ids.device)
        for i in range(n_vis, seq_len):
            for j in range(n_vis, seq_len):
                if j > i:
                    mask[i, j] = True
                    
        # BƯỚC 5: Đi qua các tầng LLM Decoder
        hidden, _ = self.layer1(multimodal_seq, mask=mask)
        hidden, attn2 = self.layer2(hidden, mask=mask)
        
        logits = self.lm_head(hidden)
        return logits, attn2, n_vis, n_text

    def count_parameters(self) -> Dict[str, int]:
        """Thống kê chi tiết số lượng tham số học được của từng khối."""
        vis_params = sum(p.numel() for p in self.vision_encoder.parameters())
        proj_params = sum(p.numel() for p in self.projector.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        llm_params = total_params - vis_params - proj_params
        return {
            "vision_encoder": vis_params,
            "projector": proj_params,
            "llm_backbone": llm_params,
            "total": total_params
        }

    def generate(self, img: torch.Tensor, prompt_ids: List[int], max_steps: int = 5, temperature: float = 1.0) -> List[Tuple[int, str, float, List[Tuple[str, float]]]]:
        """Vòng lặp sinh tự hồi quy (Autoregressive Generation) từng bước."""
        curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=img.device)
        history = []
        
        for step in range(max_steps):
            with torch.no_grad():
                logits, _, _, _ = self.forward(img, curr_ids)
                next_logits = logits[0, -1, :] / max(temperature, 1e-4)
                probs = F.softmax(next_logits, dim=-1)
                
                top_probs, top_ids = torch.topk(probs, min(3, len(VOCAB)))
                chosen_id = top_ids[0].item()
                chosen_word = ID2WORD.get(chosen_id, '<unk>')
                chosen_prob = top_probs[0].item()
                
                candidates = [(ID2WORD.get(top_ids[k].item(), '<unk>'), top_probs[k].item()) for k in range(len(top_ids))]
                history.append((step + 1, chosen_word, chosen_prob, candidates))
                
                curr_ids = torch.cat([curr_ids, torch.tensor([[chosen_id]], device=img.device)], dim=1)
                if chosen_word == '<eos>':
                    break
                    
        return history
