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
    """Kiểm thử hàm phân đoạn ảnh thành 16 patches 16x16 từ ảnh 64x64."""
    pil_img, _ = create_sample_image(size=64)
    patches = extract_patches(pil_img, patch_size=16)
    assert len(patches) == 16, f"Số patches phải là 16, thực tế: {len(patches)}"
    for idx, patch_img, box in patches:
        assert patch_img.size == (16, 16), f"Kích thước patch #{idx} không phải 16x16: {patch_img.size}"


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
