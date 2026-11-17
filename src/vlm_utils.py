"""Các tiện ích xử lý dữ liệu và công cụ hỗ trợ mô hình Vision-Language (VLM).

Bao gồm:
- Bộ từ vựng chuẩn hóa đa phương thức (VOCAB, ID2WORD).
- Tạo ảnh mẫu tổng hợp có kiểm soát (vật thể hình học, nhãn chữ, tọa độ).
- Trích xuất và phân đoạn ảnh thành các Patch (Vision Transformer).
- Tầng tinh chỉnh tham số hiệu quả (PEFT / LoRA) trên Projector.
- Dữ liệu và hàm đo lường đánh giá ảo giác (POPE Benchmark).
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


# 1. BỘ TỪ VỰNG CHUẨN HÓA CHO VLM (VOCABULARY)
VOCAB: Dict[str, int] = {
    '<pad>': 0, '<bos>': 1, '<eos>': 2, '<img_token>': 3,
    'hình': 4, 'tròn': 5, 'tam': 6, 'giác': 7, 'vuông': 8,
    'đỏ': 9, 'xanh': 10, 'vàng': 11, 'màu': 12, 'gì': 13, '?': 14,
    'ở': 15, 'đâu': 16, 'có': 17, 'không': 18, 'là': 19,
    'trên': 20, 'dưới': 21, 'phải': 22, 'trái': 23, 'đếm': 24,
    '1': 25, '2': 26, 'ngôi': 27, 'sao': 28, 'mô': 29, 'tả': 30,
    'ảnh': 31, 'tổng': 32, 'tiền': 33, '35k': 34, 'chữ': 35, 'vlm': 36
}
ID2WORD: Dict[int, str] = {idx: word for word, idx in VOCAB.items()}


def text_to_ids(text: str) -> List[int]:
    """Chuyển chuỗi văn bản thành danh sách token ID dựa trên VOCAB."""
    tokens = text.strip().lower().split()
    return [VOCAB.get(t, VOCAB['<pad>']) for t in tokens if t in VOCAB]


def ids_to_text(ids: List[int]) -> str:
    """Chuyển danh sách token ID thành chuỗi văn bản."""
    words = [ID2WORD.get(i, '<unk>') for i in ids if i not in (VOCAB['<pad>'], VOCAB['<bos>'], VOCAB['<eos>'])]
    return " ".join(words)


# 2. HÀM TẠO ẢNH MẪU TỔNG HỢP (SYNTHETIC SAMPLE IMAGE)
def create_sample_image(size: int = 64) -> Tuple[Image.Image, torch.Tensor]:
    """Tạo ảnh mẫu có kiểm soát 64x64 và Tensor PyTorch chuẩn hóa [1, 3, H, W].
    
    Cấu trúc không gian:
    - Nền: xám nhạt (245, 245, 248)
    - Vật thể 1: Hình tròn màu đỏ ở góc trên bên phải (x: 40..56, y: 8..24)
    - Vật thể 2: Hình tam giác màu xanh lá ở góc dưới bên trái (x: 8..40, y: 28..56)
    - Ký tự chữ: 'VLM'
    """
    img = Image.new("RGB", (size, size), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)
    
    # 1. Hình tròn đỏ (Góc trên bên phải)
    draw.ellipse([40, 8, 56, 24], fill=(220, 20, 60))
    
    # 2. Hình tam giác xanh lá (Góc dưới bên trái)
    draw.polygon([(8, 56), (40, 56), (24, 28)], fill=(34, 139, 34))
    
    # Chuyển đổi sang Tensor chuẩn hóa [1, 3, H, W] trong đoạn [0, 1]
    tensor_arr = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(tensor_arr).permute(2, 0, 1).unsqueeze(0)
    
    return img, img_tensor


# 3. TRÍCH XUẤT VÀ PHÂN ĐOẠN ẢNH THÀNH CÁC PATCH
def extract_patches(img_pil: Image.Image, patch_size: int = 16) -> List[Tuple[int, Image.Image, Tuple[int, int, int, int]]]:
    """Cắt ảnh thành danh sách các Patch độc lập dạng lưới N x N.
    
    Trả về danh sách các tuple: (patch_index, patch_image, (left, top, right, bottom)).
    """
    w, h = img_pil.size
    patches = []
    idx = 0
    for r in range(0, h, patch_size):
        for c in range(0, w, patch_size):
            box = (c, r, c + patch_size, r + patch_size)
            patch_img = img_pil.crop(box)
            patches.append((idx, patch_img, box))
            idx += 1
    return patches


# 4. TẦNG TINH CHỈNH THAM SỐ HIỆU QUẢ (PEFT / LORA) TRÊN PROJECTOR
class LoRALinear(nn.Module):
    """Tầng tuyến tính hỗ trợ LoRA (Low-Rank Adaptation).
    
    Đóng băng toàn bộ trọng số gốc W_0, chỉ tối ưu hai ma trận hạng thấp A và B:
    W = W_0 + (A @ B) * (alpha / rank)
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: int = 8):
        super().__init__()
        # Trọng số gốc bị đóng băng hoàn toàn (requires_grad = False)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # Nhánh thích nghi LoRA hạng thấp (requires_grad = True)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out


# 5. DỮ LIỆU ĐO LƯỜNG VÀ THỰC NGHIỆM CHUẨN
def get_pope_benchmark_data() -> Dict[str, Dict[str, float]]:
    """Dữ liệu đo lường đánh giá ảo giác trên POPE Benchmark theo 3 cấp độ bẫy."""
    return {
        "POPE-Random": {
            "accuracy": 89.2,
            "hallucination_rate": 10.8,
            "description": "Vật thể âm tính chọn ngẫu nhiên hoàn toàn từ từ điển."
        },
        "POPE-Popular": {
            "accuracy": 80.5,
            "hallucination_rate": 19.5,
            "description": "Vật thể xuất hiện với tần suất cao nhất trong tập dữ liệu tiền huấn luyện."
        },
        "POPE-Adversarial": {
            "accuracy": 68.4,
            "hallucination_rate": 31.6,
            "description": "Vật thể thường xuyên đồng xuất hiện trong ngữ cảnh tương tự nhưng thực tế vắng mặt."
        }
    }


def get_controlled_benchmark_results() -> List[Dict]:
    """Bảng 20 mẫu thử nghiệm có kiểm soát trên 4 nhóm tác vụ VLM."""
    return [
        {
            "Nhóm tác vụ": "1. Mô tả ảnh & VQA",
            "Số mẫu": 5,
            "Đúng": 5,
            "Độ chính xác (%)": 100.0,
            "Dạng lỗi phổ biến": "Không có lỗi"
        },
        {
            "Nhóm tác vụ": "2. Đọc tài liệu (DocVQA)",
            "Số mẫu": 5,
            "Đúng": 4,
            "Độ chính xác (%)": 80.0,
            "Dạng lỗi phổ biến": "Chữ số bị mờ ở biên ảnh"
        },
        {
            "Nhóm tác vụ": "3. Toán hình học & Biểu đồ",
            "Số mẫu": 5,
            "Đúng": 3,
            "Độ chính xác (%)": 60.0,
            "Dạng lỗi phổ biến": "Sai ở phép tính trừ đa bước"
        },
        {
            "Nhóm tác vụ": "4. Kiểm tra bẫy ảo giác",
            "Số mẫu": 5,
            "Đúng": 3,
            "Độ chính xác (%)": 60.0,
            "Dạng lỗi phổ biến": "Đoán mò vật thể theo thói quen ngôn ngữ (Language Prior)"
        }
    ]
