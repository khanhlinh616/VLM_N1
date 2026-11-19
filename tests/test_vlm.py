"""Bộ kiểm thử đơn vị và tích hợp cho hệ thống Vision-Language Model (VLM).

Tuân thủ Definition of Done: Code chạy, Test passed, đánh giá tự động.
"""

import sys
from pathlib import Path
import pytest
import torch

# Đảm bảo import được module trong package src/
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.vlm_model import RealisticMiniVLM, VisionEncoder, MLPProjector
from src.vlm_utils import (
    VOCAB,
    text_to_ids,
    ids_to_text,
    create_sample_image,
    extract_patches,
    LoRALinear,
    get_pope_benchmark_data,
    get_controlled_benchmark_results,
)


def test_vision_encoder():
    """Kiểm thử Vision Encoder: Tiếp nhận ảnh [B, 3, 64, 64] -> Xuất [B, 16, 64]."""
    encoder = VisionEncoder(in_channels=3, d_v=64, patch_size=16, num_patches=16)
    dummy_img = torch.randn(2, 3, 64, 64)
    features = encoder(dummy_img)
    assert features.shape == (2, 16, 64), f"Kích thước đặc trưng thị giác không khớp: {features.shape}"


def test_mlp_projector():
    """Kiểm thử Modality Projector: Chiếu đặc trưng từ d_v (64) -> d_llm (64)."""
    projector = MLPProjector(d_v=64, d_llm=64)
    dummy_features = torch.randn(2, 16, 64)
    visual_tokens = projector(dummy_features)
    assert visual_tokens.shape == (2, 16, 64), f"Kích thước visual tokens không khớp: {visual_tokens.shape}"


def test_realistic_mini_vlm_forward():
    """Kiểm thử toàn trình mô hình RealisticMiniVLM:
    - Logits shape: [B, seq_len, vocab_size]
    - Attention matrix shape: [B, seq_len, seq_len]
    - Số visual tokens = 16
    """
    model = RealisticMiniVLM(vocab_size=len(VOCAB), d_v=64, d_llm=64)
    dummy_img = torch.randn(1, 3, 64, 64)
    prompt_ids = torch.tensor([[4, 6, 7, 12, 13, 14]], dtype=torch.long)  # 'hình tam giác màu gì ?'
    
    logits, attn_weights, n_vis, n_text = model(dummy_img, prompt_ids)
    
    seq_len = n_vis + n_text
    assert n_vis == 16, f"Số visual tokens phải là 16, thực tế: {n_vis}"
    assert n_text == 6, f"Số text tokens phải là 6, thực tế: {n_text}"
    assert logits.shape == (1, seq_len, len(VOCAB)), f"Logits shape không khớp: {logits.shape}"
    assert attn_weights.shape == (1, seq_len, seq_len), f"Attention weights shape không khớp: {attn_weights.shape}"


def test_extract_patches():
    """Kiểm thử hàm phân đoạn ảnh thành 16 patches 16x16 từ ảnh 64x64 và kiểm tra vị trí vật thể."""
    pil_img, _ = create_sample_image(size=64)
    patches = extract_patches(pil_img, patch_size=16)
    assert len(patches) == 16, f"Số patches phải là 16, thực tế: {len(patches)}"
    for idx, patch_img, box in patches:
        assert patch_img.size == (16, 16), f"Kích thước patch #{idx} không phải 16x16: {patch_img.size}"

    # Kiểm tra phân bố chính xác các patch chứa từng thực thể hình học
    import numpy as np
    circle_patches = []
    triangle_patches = []
    square_patches = []
    empty_patches = []

    for idx, patch_img, _ in patches:
        arr = np.array(patch_img)
        # Hình tròn đỏ: R > 180, G < 100, B < 100
        has_circle = np.any((arr[:, :, 0] > 180) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100))
        # Hình tam giác xanh lá: G > 120, R < 100, B < 100
        has_triangle = np.any((arr[:, :, 1] > 120) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 100))
        # Hình vuông vàng: R > 150, G > 90, B < 50
        has_square = np.any((arr[:, :, 0] > 150) & (arr[:, :, 1] > 90) & (arr[:, :, 2] < 50))

        if has_circle:
            circle_patches.append(idx)
        if has_triangle:
            triangle_patches.append(idx)
        if has_square:
            square_patches.append(idx)
        if not (has_circle or has_triangle or has_square):
            empty_patches.append(idx)

    assert circle_patches == [2, 3, 6, 7], f"Hình tròn phải nằm ở các patch [2, 3, 6, 7], thực tế: {circle_patches}"
    assert triangle_patches == [5, 8, 9, 10, 12, 13, 14], f"Hình tam giác phải nằm ở các patch [5, 8, 9, 10, 12, 13, 14], thực tế: {triangle_patches}"
    assert square_patches == [10, 11, 14, 15], f"Hình vuông vàng phải nằm ở các patch [10, 11, 14, 15], thực tế: {square_patches}"
    assert empty_patches == [0, 1, 4], f"Nền trống phải nằm ở các patch [0, 1, 4], thực tế: {empty_patches}"


def test_zero_emojis_and_icons():
    """Kiểm thử tự động đảm bảo toàn bộ mã nguồn và tài liệu không chứa emoji hoặc icon."""
    import unicodedata
    import os

    forbidden_characters = []
    for root, dirs, files in os.walk(str(ROOT_DIR)):
        if any(skip in root for skip in ['.git', '.pytest_cache', '__pycache__', '.venv', 'output']):
            continue
        for f in files:
            if f.endswith(('.py', '.md', '.ipynb', '.txt', '.json', '.html')):
                file_path = os.path.join(root, f)
                with open(file_path, 'r', encoding='utf-8') as fh:
                    content = fh.read()
                for char in content:
                    code = ord(char)
                    # Chặn biến thể bộ chọn emoji, khối emoji Unicode, biểu tượng và pictograph
                    if (0xFE00 <= code <= 0xFE0F or
                        0x1F300 <= code <= 0x1FAFF or
                        0x2600 <= code <= 0x27BF):
                        forbidden_characters.append((f, hex(code), unicodedata.name(char, 'UNKNOWN')))

    assert len(forbidden_characters) == 0, f"Phát hiện icon/emoji vi phạm quy tắc: {forbidden_characters}"


def test_lora_linear_gradient_isolation():
    """Kiểm thử cơ chế đóng băng trọng số gốc và cập nhật ma trận thích nghi LoRA."""
    lora = LoRALinear(in_features=64, out_features=64, rank=4, alpha=8)
    dummy_input = torch.randn(2, 64)
    output = lora(dummy_input)
    loss = output.sum()
    loss.backward()

    # Trọng số gốc phải bị đóng băng hoàn toàn (.grad is None)
    assert lora.linear.weight.grad is None, "Trọng số gốc không được có gradient!"
    # Ma trận LoRA A và B phải nhận được gradient bình thường
    assert lora.lora_A.grad is not None, "Ma trận LoRA A phải nhận được gradient!"
    assert lora.lora_B.grad is not None, "Ma trận LoRA B phải nhận được gradient!"


def test_benchmark_data_integrity():
    """Kiểm thử tính toàn vẹn của dữ liệu benchmark POPE và bộ đo lường 20 mẫu."""
    pope_data = get_pope_benchmark_data()
    expected_splits = ["POPE-Random", "POPE-Popular", "POPE-Adversarial"]
    for s in expected_splits:
        assert s in pope_data, f"POPE phải chứa tập con {s}"
    
    controlled_results = get_controlled_benchmark_results()
    assert len(controlled_results) == 4, "Bộ kiểm thử có kiểm soát phải gồm đủ 4 nhóm tác vụ"
    total_samples = sum(item["Số mẫu"] for item in controlled_results)
    assert total_samples == 20, f"Tổng số mẫu kiểm thử phải là 20, thực tế: {total_samples}"
