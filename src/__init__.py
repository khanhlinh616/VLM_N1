"""Gói mã nguồn thực nghiệm và kiến trúc Vision-Language Model (VLM / LVM).

Các module chính:
- vlm_model: Định nghĩa kiến trúc RealisticMiniVLM, VisionEncoder, MLPProjector, VLMDecoderLayer.
- vlm_utils: Các hàm tiện ích xử lý ảnh mẫu, trích xuất patch, LoRALinear, từ vựng và benchmark POPE.
"""

from .vlm_model import RealisticMiniVLM, VisionEncoder, MLPProjector, VLMDecoderLayer
from .vlm_utils import (
    VOCAB,
    ID2WORD,
    text_to_ids,
    ids_to_text,
    create_sample_image,
    extract_patches,
    LoRALinear,
    get_pope_benchmark_data,
    get_controlled_benchmark_results,
)

__all__ = [
    "RealisticMiniVLM",
    "VisionEncoder",
    "MLPProjector",
    "VLMDecoderLayer",
    "VOCAB",
    "ID2WORD",
    "text_to_ids",
    "ids_to_text",
    "create_sample_image",
    "extract_patches",
    "LoRALinear",
    "get_pope_benchmark_data",
    "get_controlled_benchmark_results",
]
