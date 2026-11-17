# Nghiên cứu & Thực nghiệm: Mô hình Thị giác Lớn (Large Vision Models — LVM / VLM)

Gói tài liệu và mã nguồn thực nghiệm chuyên sâu phục vụ nghiên cứu các nguyên lý cốt lõi, kiến trúc nền tảng và cơ chế vận hành của **Mô hình Thị giác Lớn (Large Vision Models — LVM)** cùng nhánh ứng dụng tương tác đa phương thức tiêu biểu là **Mô hình Thị giác - Ngôn ngữ Lớn (Vision-Language Models — VLM / LVLM)**.

Dự án cung cấp bộ tài liệu và mã nguồn hoàn chỉnh gồm:
- **Tài liệu nghiên cứu chuyên sâu:** [docs/VLM.md](file:///d:/Hoc_Sau/VLM/docs/VLM.md) — Phân tích chi tiết 10 chuyên đề lý thuyết, công thức toán học và kiến trúc kỹ thuật.
- **Notebook thực nghiệm toàn trình:** [notebook/vlm_demo.ipynb](file:///d:/Hoc_Sau/VLM/notebook/vlm_demo.ipynb) — Mô phỏng trực quan từ khâu cắt Patch ảnh, chiếu đặc trưng (Projector), tính toán ma trận Attention xuyên phương thức thật (Cross-Modal Attention Heatmap), vòng lặp sinh tự hồi quy từng bước đến đánh giá ảo giác qua POPE Benchmark.
- **Dữ liệu đo lường đầu ra:** [output/vlm_benchmark_results.csv](file:///d:/Hoc_Sau/VLM/output/vlm_benchmark_results.csv) — Kết quả đo lường định lượng trên các tập mẫu thử nghiệm.
- **Mã nguồn hệ thống:** [src/](file:///d:/Hoc_Sau/VLM/src) — Toàn bộ các script thực thi và module thuật toán.

---

## 1. Cấu trúc thư mục dự án

```
d:/Hoc_Sau/VLM/
├── docs/
│   └── VLM.md                    Tài liệu nghiên cứu chuyên sâu 10 chuyên đề kỹ thuật LVM / VLM
├── notebook/
│   └── vlm_demo.ipynb            Notebook mô phỏng toàn trình và trực quan hóa thực nghiệm
├── output/
│   ├── README.md                 Mô tả các kết quả và artifact đo lường
│   └── vlm_benchmark_results.csv Bảng dữ liệu định lượng đo lường độ chính xác và phân loại lỗi
├── src/
│   ├── __init__.py               Khởi tạo package và export các module VLM cốt lõi
│   ├── vlm_model.py              Kiến trúc RealisticMiniVLM, VisionEncoder, MLPProjector, VLMDecoderLayer
│   ├── vlm_utils.py              Tiện ích trích xuất patch ảnh, từ vựng đa phương thức, LoRALinear, POPE benchmark
│   └── run_vlm_experiment.py     Script thực nghiệm suy luận toàn trình, đo lường định lượng và xuất kết quả
├── .gitignore                    Cấu hình loại trừ Git
└── README.md                     Tài liệu tổng quan dự án
```

---

## 2. Hệ thống 10 Chuyên đề Nghiên cứu Cốt lõi

Toàn bộ lý thuyết và thực nghiệm được chuẩn hóa đồng bộ theo 10 chuyên đề kỹ thuật:

| # | Chuyên đề nghiên cứu | Nội dung trọng tâm | Minh họa & Trực quan hóa trong Notebook |
|---|---|---|---|
| **1** | **Giới thiệu về LVM / VLM** | Phân biệt Computer Vision (CV), LLM và LVM/VLM. Vai trò phá vỡ rào cản đơn thức (Symbol Grounding). | Trực quan hóa Card Layout 4 tác vụ: Captioning, VQA, Đọc tài liệu (DocVQA) và Visual Grounding. |
| **2** | **Nền tảng kiến trúc** | Phân tích 3 trụ cột: Vision Encoder (ViT/SigLIP), Modality Projector (MLP) và LLM Backbone. | Xây dựng mạng `RealisticMiniVLM` chuẩn PyTorch; thống kê phân bổ tham số giữa 3 khối. |
| **3** | **Xử lý dữ liệu ảnh & chữ** | Preprocessing, Patch Extraction ($P \times P$), Dynamic Resolution (AnyRes) và Tokenization văn bản. | Cắt ảnh thật thành lưới 16 patches độc lập ($4 \times 4$), phân tích vector pixel và tokenize prompt. |
| **4** | **Nguyên lý hoạt động** | Luồng tính toán 4 bước: Image/Text Encoding $\to$ Alignment $\to$ Multimodal Fusion qua Self-Attention. | Chạy Forward Pass thật, trích xuất ma trận Attention thật và vẽ Heatmap từ ngữ chiếu lên từng Patch. |
| **5** | **Các giai đoạn huấn luyện** | Pretraining (CLIP Contrastive InfoNCE Loss), Image-Text Matching, Visual SFT và Alignment (DPO/RLHF). | Huấn luyện đối sánh trên 3 cặp (ảnh, text) thật, tính ma trận Cosine Similarity và vẽ biểu đồ nhiệt. |
| **6** | **Quá trình sinh phản hồi** | Cơ chế sinh tự hồi quy có Visual Prefix, vai trò của KV Cache và các chiến lược giải mã ($T$, top-$p$, greedy). | Vòng lặp sinh từng bước (Step-by-step Loop), in xác suất từng từ và ảnh hưởng của Temperature. |
| **7** | **Khả năng của LVM / VLM** | Đánh giá 5 năng lực: VQA, Image Captioning, OCR không cần module phụ, Visual Reasoning, Visual Grounding. | Vẽ trực tiếp Bounding Box tọa độ lên ảnh thật, giải toán hình học và trích xuất số liệu. |
| **8** | **Thích nghi & Ứng dụng** | Prompting & ICL, Full Fine-tuning, Parameter-Efficient Fine-Tuning (PEFT/LoRA) và Multimodal RAG. | Cài đặt tầng `LoRALinear`, chạy forward/backward thật, kiểm tra gradient (tiết kiệm >95% VRAM). |
| **9** | **Hạn chế & Đánh giá ảo giác**| Object Hallucination (do Language Prior lấn át Visual Features), giới hạn độ phân giải. POPE Benchmark. | Thử nghiệm bẫy ảo giác trên vật thể không tồn tại trong ảnh, vẽ biểu đồ POPE 3 mức bẫy đối kháng. |
| **10**| **Demo & Kết luận thực nghiệm**| Thực nghiệm có kiểm soát trên bộ 20 mẫu, lập ma trận nhầm lẫn, phân tích lỗi sai và bài học rút ra. | Bảng dữ liệu 20 mẫu, biểu đồ cột độ chính xác theo nhóm, biểu đồ tròn phân loại 3 nguyên nhân lỗi. |

---

## 3. Cài đặt và Môi trường thực thi

Yêu cầu môi trường Python 3.8 trở lên với các thư viện Deep Learning tiêu chuẩn:

```bash
# Cài đặt các thư viện cần thiết
pip install torch torchvision numpy pandas matplotlib pillow ipykernel nbformat
```

### Cách 1: Chạy trực tiếp Notebook trực quan (Khuyến nghị)
Mở tệp [notebook/vlm_demo.ipynb](file:///d:/Hoc_Sau/VLM/notebook/vlm_demo.ipynb) trong Visual Studio Code hoặc Jupyter Lab và chọn **Run All**. Tất cả các ô code đã được tối ưu hóa để chạy mượt mà trên cả CPU và GPU dân dụng mà không gặp bất kỳ lỗi nào.

### Cách 2: Thực thi tự động và xuất lại kết quả
```bash
jupyter nbconvert --to notebook --execute notebook/vlm_demo.ipynb --output notebook/vlm_demo.ipynb
```

### Cách 3: Chạy script thực nghiệm suy luận từ thư mục src/
```bash
# Thực thi thực nghiệm toàn trình với cấu hình tiêu chuẩn:
python src/run_vlm_experiment.py

# Tùy chỉnh tham số nhiệt độ giải mã (temperature) và số bước tự hồi quy:
python src/run_vlm_experiment.py --temperature 0.7 --steps 4 --export-csv output/vlm_benchmark_results.csv
```

---

## 4. Tổng hợp Kết quả Thực nghiệm Chính

| Nội dung khảo sát | Kết quả kỹ thuật ghi nhận trong thực nghiệm |
|---|---|
| **Phân đoạn Patch (Mục 3)** | Ảnh $64 \times 64$ được phân tách chuẩn xác thành lưới $4 \times 4 = 16$ patches ($16 \times 16$ điểm ảnh). Mỗi patch tương đương vector 768 giá trị số thực trước khi chiếu tuyến tính vào không gian embedding $\mathbb{R}^{64}$. |
| **Cross-Modal Attention (Mục 4)** | Ma trận Attention trích xuất từ tầng MultiheadAttention chứng minh các token văn bản như *"hình tam giác"* tập trung trọng số kích hoạt cao nhất vào các Patch 9, 10, 13 và 14 (vùng không gian chứa hình tam giác xanh). |
| **Học đối sánh Contrastive (Mục 5)** | Hàm mất mát InfoNCE Loss hội tụ trên batch 3 cặp ảnh-chữ với ma trận tương đồng Cosine đạt giá trị cực đại trên đường chéo (cặp đúng), thể hiện khả năng căn chỉnh không gian giữa Vision và Language. |
| **Chiến lược giải mã (Mục 6)** | Greedy decoding ($T \to 0$) đảm bảo tính xác định cao cho các bài toán OCR và giải toán; Temperature $T=0.7 - 0.8$ cân bằng tốt nhất giữa tính mạch lạc cú pháp và độ phong phú từ vựng khi mô tả ảnh tự do. |
| **Tối ưu hóa LoRA (Mục 8)** | Gắn các ma trận hạng thấp ($r=4$) vào tầng Projector giúp đóng băng 100% trọng số gốc (`.grad = None`), giảm hơn **95%** số lượng tham số cần huấn luyện và giải phóng áp lực bộ nhớ VRAM. |
| **Ảo giác vật thể (Mục 9 & 10)** | Đánh giá trên POPE Benchmark cho thấy tỷ lệ ảo giác tăng dần khi bẫy đối kháng tinh vi hơn (từ 10.8% ở Random lên 31.6% ở Adversarial), bắt nguồn từ việc LLM prior tự động "điền vào chỗ trống" các vật thể quen thuộc dù ảnh không có. |

---

## 5. Tài liệu Tham khảo Học thuật

1. **Dosovitskiy et al. (2020)** — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)*. ICLR 2021.
2. **Radford et al. (2021)** — *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML 2021.
3. **Liu et al. (2023)** — *Visual Instruction Tuning (LLaVA)*. NeurIPS 2023.
4. **Wang et al. (2023)** — *Sequential Modeling Enables Scalable Learning for Large Vision Models (LVM)*. arXiv:2312.00785.
5. **Wang et al. (2024)** — *Qwen2-VL: To See the World More Clearly*. arXiv:2409.12191.
6. **Li et al. (2023)** — *Evaluating Object Hallucination in Large Vision-Language Models (POPE)*. EMNLP 2023.
