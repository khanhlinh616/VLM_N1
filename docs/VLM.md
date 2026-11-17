# Nghiên cứu Chuyên sâu: Mô hình Thị giác Lớn (Large Vision Models — LVM / VLM)

## Dàn ý cấu trúc kỹ thuật tổng quan

| # | Mục | Cơ sở lý thuyết & Nguyên lý cốt lõi | Minh họa kỹ thuật & Tác vụ thực nghiệm |
|---|---|---|---|
| 1 | **Giới thiệu** | Mô hình Thị giác Lớn (LVM / VLM) là gì? Phân biệt Computer Vision, LLM và VLM. Vai trò của mô hình đa phương thức trong AI hiện đại. | Bảng so sánh 3 mô hình & ví dụ 4 tác vụ: mô tả ảnh, hỏi đáp hình ảnh, đọc tài liệu, nhận diện nội dung. |
| 2 | **Nền tảng kiến trúc** | Các thành phần chính: Vision Encoder, Projector/Adapter và Language Model. Cách kết nối thông tin thị giác với ngôn ngữ. | Sơ đồ kiến trúc VLM toàn diện từ hình ảnh và văn bản đầu vào đến câu trả lời đầu ra. |
| 3 | **Xử lý dữ liệu hình ảnh và văn bản** | Image Preprocessing, Image Patches/Crops, Tokenization và Embedding. Cách chuyển đổi hình ảnh và văn bản thành biểu diễn số học mà mô hình có thể xử lý. | Minh họa quá trình chia ảnh thành patches (lưới 2D) và chuyển văn bản thành token (subword token ID). |
| 4 | **Nguyên lý hoạt động** | Image Encoding, Text Encoding, Multimodal Fusion và Vision-Language Alignment. Cách mô hình liên kết nội dung hình ảnh với ngôn ngữ. | Minh họa từng bước luồng tính toán khi đưa một hình ảnh kèm prompt cụ thể để tạo câu trả lời. |
| 5 | **Các giai đoạn huấn luyện** | Vision-Language Pretraining, Image-Text Contrastive Learning, Image-Text Matching và Multimodal Instruction Tuning (Visual SFT / Alignment). | Sơ đồ mục tiêu, dữ liệu đầu vào và kết quả thu được của từng giai đoạn huấn luyện. |
| 6 | **Quá trình sinh phản hồi** | Cách VLM tiếp nhận hình ảnh và prompt, xử lý thông tin đa phương thức và sinh phản hồi. Vai trò của Autoregressive Generation và Decoding strategies. | Cùng một hình ảnh thực tế với các prompt khác nhau và kết quả phản hồi tương ứng. |
| 7 | **Khả năng của VLM** | Visual Question Answering (VQA), Image Captioning, OCR, Visual Reasoning và Document Understanding. | Thử nghiệm và đánh giá nhiều tác vụ chuyên sâu trên các hình ảnh thực tế. |
| 8 | **Cách thích nghi và ứng dụng VLM** | Prompting, Fine-tuning toàn phần, Parameter-Efficient Fine-Tuning (PEFT/LoRA) và Multimodal Retrieval-Augmented Generation (Multimodal RAG). | Bảng so sánh phương pháp: dữ liệu cần có, chi phí tài nguyên, ưu/nhược điểm và trường hợp sử dụng. |
| 9 | **Hạn chế và đánh giá** | Hallucination (ảo giác vật thể), hạn chế nhận diện chi tiết, độ phân giải hình ảnh, thiên lệch dữ liệu và giới hạn suy luận thị giác. Các phương pháp đánh giá chất lượng VLM. | Một ví dụ mô hình mô tả sai hình ảnh thực tế, phân tích nguyên nhân gốc rễ và quy trình kiểm chứng. |
| 10 | **Demo và kết luận** | Thực nghiệm VLM trên các tác vụ đã chọn, tổng hợp kết quả, đánh giá năng lực thực tế, phân tích ma trận nhầm lẫn và lỗi sai. | Bảng kết quả thực nghiệm, hình ảnh đầu vào, câu trả lời đầu ra và phân tích chi tiết các ca lỗi. |

---

## 1. Giới thiệu

### 1.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

#### a. Mô hình Thị giác Lớn (LVM / VLM) là gì?
**LVM (Large Vision Model)** — trong bối cảnh tương tác đa phương thức thường được triển khai dưới dạng **Mô hình Thị giác - Ngôn ngữ Lớn (Large Vision-Language Model - LVLM / VLM)** hoặc **Mô hình Thị giác Tự hồi quy (Sequential LVM)** — là lớp mô hình trí tuệ nhân tạo có khả năng tiếp nhận, xử lý và suy luận đồng thời trên cả hai phương thức dữ liệu: **Thị giác** (Hình ảnh, Video) và **Ngôn ngữ** (Văn bản). 

Khác với các hệ thống AI truyền thống chỉ làm việc trên một phương thức đơn lẻ, VLM sử dụng ngôn ngữ tự nhiên làm "giao diện tư duy chung" để hiểu sâu sắc nội dung hình ảnh, trả lời câu hỏi, suy luận logic và thực hiện chỉ dẫn của con người.

#### b. Phân biệt Computer Vision (CV), LLM và VLM
Quá trình phát triển của trí tuệ nhân tạo trải qua ba bước chuyển dịch lớn:
1. **Computer Vision truyền thống (CV):** Mô hình hóa dữ liệu ảnh dựa trên các mạng tích chập (CNN) như ResNet hoặc Vision Transformer (ViT). Đầu ra bị giới hạn trong không gian nhãn đóng cố định (closed-set), ví dụ: 1000 nhãn phân loại của ImageNet, 80 lớp phát hiện của COCO. Mô hình chỉ biết "nhìn" và gán nhãn, không thể đàm thoại hay giải thích ngữ cảnh.
2. **Mô hình Ngôn ngữ Lớn (LLM):** Điển hình là GPT-3/4, LLaMA, Qwen. Mô hình được huấn luyện trên hàng nghìn tỷ token văn bản, sở hữu năng lực suy luận và vốn tri thức khổng lồ nhưng hoàn toàn "mù" đối với thế giới vật lý thực tế.
3. **Mô hình Thị giác - Ngôn ngữ (VLM):** Kết hợp sức mạnh thị giác của CV với bộ não tư duy của LLM. VLM mở rộng không gian đầu ra thành ngôn ngữ tự do (open-ended natural language), cho phép con người đặt bất kỳ câu hỏi nào về bức ảnh.

| Tiêu chí | Computer Vision truyền thống (CV) | Mô hình Ngôn ngữ Lớn (LLM) | Mô hình Thị giác - Ngôn ngữ (VLM) |
|---|---|---|---|
| **Dữ liệu đầu vào** | Chỉ có Hình ảnh / Video | Chỉ có Văn bản (Text/Code) | **Cả Hình ảnh/Video VÀ Văn bản** |
| **Dữ liệu đầu ra** | Nhãn phân loại, Bounding box, Mask | Chuỗi văn bản tiếp theo | **Văn bản phân tích, tọa độ vật thể, lời giải** |
| **Không gian nhãn** | Không gian đóng (Closed-set: tập nhãn định sẵn) | Không gian mở (Open-ended: từ vựng văn bản) | **Không gian mở đa phương thức** |
| **Cơ chế suy luận** | Trích xuất đặc trưng hình học/màu sắc | Dự đoán token tiếp theo (Next-token prediction) | **Suy luận ngôn ngữ có điều kiện thị giác** |
| **Ví dụ tiêu biểu** | ResNet-50, YOLOv8, Mask R-CNN | GPT-3.5, LLaMA-3, Qwen-2.5 | **LLaVA, Qwen2-VL, BLIP-2, GPT-4o** |

#### c. Vai trò của mô hình đa phương thức trong AI hiện đại
Thế giới thực của con người vốn mang tính chất đa phương thức (multimodal): chúng ta nhìn hình ảnh, đọc văn bản, lắng nghe âm thanh để đưa ra quyết định.
- **Phá vỡ giới hạn đơn thức:** VLM giải quyết vấn đề "Symbol Grounding Problem" trong NLP — giúp các từ ngữ trừu tượng trong LLM được liên kết trực tiếp với các thực thể hình ảnh ngoài đời thực.
- **Trở thành mắt thần cho AI:** Cho phép tự động hóa các tác vụ phức tạp: hỗ trợ người khiếm thị, phân tích hồ sơ bệnh án kết hợp phim chụp X-quang, đọc hiểu tự động hóa đơn/chứng từ kế toán, và làm bộ não nhận thức cho Robot tự hành (Embodied AI).

---

### 1.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Bốn tác vụ cốt lõi của VLM

Bảng minh họa các bài toán thực tế mà VLM giải quyết vượt trội so với các hệ thống đơn phương thức:

| Tác vụ | Hình ảnh đầu vào minh họa | Câu hỏi / Prompt | Đầu ra thực tế của VLM |
|---|---|---|---|
| **1. Mô tả ảnh (Image Captioning)** | Ảnh chụp một góc phố cổ Hà Nội có gánh hàng hoa, xe xích lô và người đi bộ trong một sáng mùa thu. | *"Hãy mô tả chi tiết khung cảnh trong bức ảnh này."* | *"Bức ảnh chụp một góc phố cổ rợp bóng cây vào buổi sáng. Phía bên phải là người bán hoa tươi trên chiếc xe đạp cũ, ở giữa là một người đạp xích lô đang chở khách du lịch, toát lên vẻ thanh bình đặc trưng của Hà Nội."* |
| **2. Hỏi đáp hình ảnh (Visual QA - VQA)** | Ảnh chụp bảng vi mạch điện tử với nhiều tụ điện, chip vi xử lý và các mối hàn. | *"Tụ điện lớn màu xanh dương nằm ở góc nào và có dấu hiệu phồng rộp không?"* | *"Tụ điện lớn màu xanh nằm ở góc trên bên trái của bo mạch. Bề mặt trên của tụ phẳng, không có dấu hiệu phồng rộp hay rò rỉ dung dịch điện phân."* |
| **3. Đọc hiểu tài liệu (DocVQA / OCR)** | Ảnh chụp hóa đơn thanh toán nhà hàng bị mờ một góc và hơi nghiêng. | *"Tổng số tiền thanh toán là bao nhiêu và thuế VAT được tính mấy phần trăm?"* | *"Tổng số tiền thanh toán cuối cùng là 850.000 VNĐ. Hóa đơn ghi rõ thuế VAT là 8% (tương đương 62.963 VNĐ) đã được cộng vào tổng bill."* |
| **4. Nhận diện nội dung & Tọa độ (Grounding)** | Ảnh chụp phòng khách gia đình có bàn trà, ghế sofa, tivi và một chú mèo. | *"Xác định vị trí chú mèo đang ngủ và chiếc điều khiển tivi."* | `[{"label": "chú mèo", "box_2d": [620, 310, 780, 450]}, {"label": "điều khiển tivi", "box_2d": [540, 520, 580, 610]}]` |

---

## 2. Nền tảng kiến trúc

### 2.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

Một mô hình VLM tiêu chuẩn hiện đại gồm 3 thành phần kiến trúc trụ cột:

```
[ Hình ảnh ] ──► [ Vision Encoder ] ──► [ Projector / Adapter ] ──┐
                                                                 ├──► [ Language Model (LLM) ] ──► [ Câu trả lời ]
[ Văn bản  ] ──────────────────────────► [ Text Tokenizer ] ─────┘
```

#### a. Vision Encoder (Bộ mã hóa thị giác)
- **Nhiệm vụ:** Tiếp nhận ảnh thô dưới dạng lưới pixel $X \in \mathbb{R}^{H \times W \times C}$, chia ảnh thành các mẩu nhỏ (patches) và trích xuất các vector đặc trưng không gian sâu $Z_v \in \mathbb{R}^{N_v \times D_v}$.
- **Các kiến trúc phổ biến:**
  - *CLIP-ViT (OpenAI):* Huấn luyện đối sánh ảnh-văn bản, đặc trưng rất giàu ngữ nghĩa khái niệm.
  - *SigLIP (Google):* Tối ưu hóa hàm mất mát Sigmoid giúp nhận diện các chi tiết văn bản và đối tượng nhỏ tốt hơn.
  - *EVA-CLIP / DINOv2:* Cung cấp đặc trưng biểu diễn cấu trúc không gian và hình học sắc nét.

#### b. Projector / Adapter (Bộ kết nối đa phương thức)
- **Vấn đề kỹ thuật:** Không gian vector của Vision Encoder ($D_v$, thường là 768 hoặc 1024) không cùng số chiều và không cùng không gian ngữ nghĩa với LLM Backbone ($D_{\text{llm}}$, thường là 4096 hoặc 5120).
- **Giải pháp:** Projector đóng vai trò cầu nối "phiên dịch" các vector thị giác thành các **Visual Tokens** $H_v \in \mathbb{R}^{N_v \times D_{\text{llm}}}$ để LLM hiểu được:
  - *Linear / MLP Projector (LLaVA):* Dùng 2 lớp Fully-Connected kèm kích hoạt GELU. Ưu điểm: Đơn giản, giữ nguyên toàn bộ thông tin không gian vị trí.
  - *Q-Former (BLIP-2):* Dùng cơ chế Cross-Attention với một số lượng cố định các Learnable Query Tokens (ví dụ 32 tokens) để "nén" thông tin ảnh, giảm độ dài chuỗi cho LLM.
  - *Perceiver Resampler (Flamingo):* Nén số lượng token ảnh tùy ý về kích thước cố định bằng cơ chế chú ý lặp.

#### c. Language Model Backbone (Bộ não ngôn ngữ)
- **Nhiệm vụ:** Tiếp nhận toàn bộ chuỗi token hỗn hợp gồm các Visual Tokens và Text Tokens.
- Thường là các mô hình Transformer Decoder-only mã nguồn mở mạnh mẽ nhất hiện nay như: LLaMA-3, Qwen-2.5, Vicuna, Mistral.
- LLM xem các token thị giác như một "ngôn ngữ lạ" đã được dịch sẵn vào không gian từ vựng, từ đó tiến hành tính toán Attention và suy luận logic để sinh phản hồi.

---

### 2.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Sơ đồ kiến trúc VLM toàn diện

```
                             KIẾN TRÚC TỔNG THỂ CỦA VLM (LLaVA-STYLE)
                             
  [ HÌNH ẢNH ĐẦU VÀO ]                                        [ VĂN BẢN ĐẦU VÀO / PROMPT ]
      (336 x 336 x 3)                                              "Bức ảnh có con gì?"
             │                                                              │
             ▼                                                              ▼
   ┌───────────────────┐                                          ┌───────────────────┐
   │ Chia thành Lưới   │                                          │  Text Tokenizer   │
   │  24 x 24 Patches  │                                          │  (BPE / Sentence) │
   │ (mỗi patch 14x14) │                                          └───────────────────┘
   └───────────────────┘                                                    │
             │  576 patches                                                 ▼
             ▼                                                    [ Token IDs Văn bản ]
   ┌───────────────────┐                                             [1205, 432, 98]
   │  Vision Encoder   │                                                    │
   │ (CLIP-ViT-L/14)   │                                                    ▼
   └───────────────────┘                                          ┌───────────────────┐
             │  Đặc trưng thị giác: Z_v ∈ ℝ^{576 × 1024}          │ Embedding Layer   │
             ▼                                                    └───────────────────┘
   ┌───────────────────┐                                                    │
   │ Modality Adapter  │                                                    ▼
   │  (2-layer MLP)    │                                          [ Text Tokens H_t ]
   └───────────────────┘                                              ∈ ℝ^{N_t × 4096}
             │  Visual Tokens: H_v ∈ ℝ^{576 × 4096}                         │
             └───────────────────────────────┬──────────────────────────────┘
                                             ▼
                      CHUỖI ĐẦU VÀO HỖN HỢP GHÉP NỐI (CONCATENATION)
                      [ H_v^{(1)}, H_v^{(2)}, ..., H_v^{(576)}, H_t^{(1)}, ..., H_t^{(N_t)} ]
                                             │
                                             ▼
                        ┌────────────────────────────────────────┐
                        │      LLM BACKBONE (Decoder-Only)       │
                        │      (Qwen-2.5 / LLaMA-3 / Vicuna)     │
                        │  Causal Self-Attention qua N Layers    │
                        └────────────────────────────────────────┘
                                             │
                                             ▼
                                    [ Lớp Softmax Đầu Ra ]
                                             │
                                             ▼
                               [ SINH CÂU TRẢ LỜI TỰ NHIÊN ]
                               "Trong ảnh là một chú mèo mướp."
```

---

## 3. Xử lý dữ liệu hình ảnh và văn bản

### 3.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

#### a. Tiền xử lý hình ảnh (Image Preprocessing)
Mô hình deep learning không thể tiếp nhận trực tiếp ảnh có kích thước bất kỳ. Quy trình xử lý gồm:
1. **Resize / Padding:** Đưa ảnh về kích thước chuẩn mà Vision Encoder yêu cầu (ví dụ $336 \times 336$ hoặc $448 \times 448$), bảo toàn tỷ lệ bằng cách thêm dải đen (letterbox padding).
2. **Chuẩn hóa giá trị Pixel (Normalization):** Chuyển dải màu $[0, 255]$ về đoạn $[0.0, 1.0]$, sau đó trừ trung bình (mean) và chia độ lệch chuẩn (std) theo chuẩn ImageNet:
   $$x_{\text{norm}} = \frac{x / 255.0 - \mu}{\sigma}, \quad \mu = [0.485, 0.456, 0.406], \; \sigma = [0.229, 0.224, 0.225]$$

#### b. Image Patches và Trích xuất lát cắt (Crops / AnyRes)
- **Patch Extraction (ViT):** Ảnh kích thước $(H, W, C)$ được chia thành các ô vuông kích thước $P \times P$. Số lượng patches $N = \frac{H \cdot W}{P^2}$. Mỗi patch được làm phẳng thành một vector chiều $P^2 \cdot C$.
- **Kỹ thuật Dynamic Crops (AnyRes / Dynamic Resolution):** Khi gặp ảnh độ phân giải cao (ví dụ văn bản tài liệu $1920 \times 1080$), nếu co về $336 \times 336$ chữ sẽ bị vỡ vụn. Các VLM hiện đại chia ảnh lớn thành nhiều mẩu con (crops/tiles) kích thước $336 \times 336$ để xử lý chi tiết, kết hợp một ảnh thu nhỏ toàn cục (overview thumbnail) để giữ ngữ cảnh.

#### c. Tokenization và Biểu diễn số học (Embedding)
- **Văn bản:** Được xử lý bằng thuật toán Subword (như BPE hoặc SentencePiece), biến câu chữ thành chuỗi các Token ID số nguyên $T = [t_1, t_2, \dots, t_m]$, sau đó tra bảng ma trận Embedding $E_{\text{text}}$ để lấy vector biểu diễn chiều $D$.
- **Hình ảnh:** Mỗi patch được chiếu tuyến tính qua ma trận $W_p$, sau đó cộng thêm **2D Spatial Positional Embedding** để mô hình biết được patch nào nằm ở trên, dưới, trái, phải trong bức ảnh.

---

### 3.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Quá trình phân đoạn Patch và Token hóa

#### a. Minh họa quá trình chia ảnh thành Patches
Giả sử có bức ảnh kích thước $336 \times 336$ pixel, kích thước patch $P = 14 \times 14$:

```
     Kích thước ảnh gốc: 336 x 336 pixel
     ┌───────┬───────┬───────┬───┬───────┐  ▲
     │Patch 1│Patch 2│Patch 3│...│Patch24│  │ 24 patches
     ├───────┼───────┼───────┼───┼───────┤  │ theo chiều dọc
     │Patch25│Patch26│   .   │   │       │  │ (336 / 14 = 24)
     ├───────┼───────┼───────┼───┼───────┤  │
     │   .   │   .   │   .   │   │       │  │
     ├───────┼───────┼───────┼───┼───────┤  │
     │Patch553 ...                   576 │  ▼
     └───────┴───────┴───────┴───┴───────┘
     ◄────────── 24 patches ─────────────►
          (theo chiều ngang: 336 / 14 = 24)

  Tổng số Patches: 24 × 24 = 576 patches.
  Mỗi Patch thô: Ma trận (14 × 14 × 3) = 588 giá trị số thực.
         │
         ▼ (Làm phẳng + Linear Projection + Positional Embedding)
  576 Visual Vectors, mỗi vector có kích thước D_v = 1024.
```

#### b. Minh họa Tokenize câu văn bản Prompt
Prompt tiếng Việt: `"Mô tả bức ảnh này."`

| Từ / Cụm từ | Token Subword tách ra | Token ID | Ý nghĩa biểu diễn số học |
|---|---|---|---|
| Mô tả | `Mô`, `_tả` | `[1045, 8741]` | Vector biểu diễn khái niệm yêu cầu diễn giải |
| bức ảnh | `_bức`, `_ảnh` | `[15320, 4211]` | Vector biểu diễn đối tượng thị giác |
| này | `_này` | `[1120]` | Vector chỉ định đối tượng trong ngữ cảnh |
| . | `.` | `[13]` | Dấu chấm kết thúc câu |

Toàn bộ 6 text tokens này được ánh xạ thành 6 vector chiều $D_{\text{llm}} = 4096$, sẵn sàng nối đuôi 576 visual tokens.

---

## 4. Nguyên lý hoạt động

### 4.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

#### a. Bốn bước vận hành cốt lõi của VLM
1. **Image Encoding:** Ảnh được đưa qua Vision Encoder để biến đổi dữ liệu không gian điểm ảnh thành các vector đặc trưng trừu tượng mang tính ngữ nghĩa cao.
2. **Text Encoding:** Câu hỏi của người dùng được đưa qua Tokenizer và lớp Text Embedding để biến thành chuỗi các vector biểu diễn ngữ nghĩa ngôn ngữ.
3. **Vision-Language Alignment (Căn chỉnh không gian):** Các vector ảnh được truyền qua Modality Projector. Projector điều chỉnh lại tọa độ vector sao cho khái niệm thị giác (ví dụ: vùng pixel chứa hình dáng con chó) nằm cùng vùng lân cận với token văn bản `"chó"` trong không gian biểu diễn chung của LLM.
4. **Multimodal Fusion (Hợp nhất đa phương thức):** Các visual tokens và text tokens được xếp tuần tự vào cùng một chuỗi đầu vào. Mô hình sử dụng cơ chế **Self-Attention** để cho phép từng từ ngữ trong câu hỏi "soi chiếu" và tính toán trọng số tương tác trực tiếp lên từng vùng điểm ảnh liên quan.

---

### 4.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Luồng xử lý một bức ảnh cụ thể kết hợp prompt

**Tình huống thực tế:**
- **Ảnh đầu vào:** Ảnh chụp một chú mèo mướp đang nằm ngủ trên cuốn sách bên cạnh cửa sổ có nắng chiếu.
- **Prompt của người dùng:** *"Con mèo đang nằm ở đâu?"*

```
BƯỚC 1: XỬ LÝ ĐA PHƯƠNG THỨC SONG SONG
  [ẢNH CHÚ MÈO] ──► ViT ──► Projector ──► 576 Visual Tokens: [V_1, V_2, ..., V_576]
  [PROMPT CHỮ ] ──► Tokenizer ─────────► 6 Text Tokens:     ["Con", "mèo", "đang", "nằm", "ở", "đâu?"]

BƯỚC 2: HỢP NHẤT VÀ TÍNH TOÁN CAUSAL ATTENTION
  Chuỗi token nạp vào LLM:
  [ V_1, V_2, ... V_576,  "Con", "mèo", "đang", "nằm", "ở", "đâu?" ]
                             │      │             │
                             │      └─────────────┼────────────────────────┐
                             ▼                    ▼                        ▼
       Self-Attention kết nối từ "mèo"      Self-Attention kết nối     Trọng số Attention
       với các visual tokens vùng giữa      từ "nằm ở đâu" với vùng    tập trung cao nhất
       bức ảnh (nơi có hoa văn lông mèo)    bên dưới (vùng cuốn sách)  vào 2 vùng ảnh này

BƯỚC 3: DỰ ĐOÁN TOKEN TIẾP THEO (NEXT-TOKEN PREDICTION)
  - Vị trí cuối cùng xuất ra phân phối xác suất trên toàn bộ từ vựng:
      P("Con")    = 85%
      P("Nó")     = 10%
      P("Bức ảnh")= 3%
  - Token "Con" được chọn. Tiếp tục đưa "Con" vào chuỗi để sinh tiếp tự hồi quy:
      "Con" ──► "mèo" ──► "đang" ──► "nằm" ──► "ngủ" ──► "trên" ──► "cuốn" ──► "sách" ──► "."
```

---

## 5. Các giai đoạn huấn luyện VLM

### 5.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

Huấn luyện một VLM hoàn chỉnh thường trải qua quy trình nghiêm ngặt gồm 3 giai đoạn:

```
[ Giai đoạn 1: Vision-Language Pretraining ] ──► Căn chỉnh Projector (Đóng băng Vision & LLM)
                      │
                      ▼
[ Giai đoạn 2: Multimodal Instruction Tuning ] ──► Dạy làm theo chỉ dẫn (Mở khóa LLM + Projector)
                      │
                      ▼
[ Giai đoạn 3: Multimodal Preference Alignment ] ──► Triệt tiêu ảo giác (DPO / RLHF)
```

#### a. Giai đoạn 1: Vision-Language Pretraining (VLP - Tiền huấn luyện căn chỉnh đặc trưng)
- **Ý tưởng:** Khi mới ghép nối, Vision Encoder và LLM nói hai "ngôn ngữ vector" hoàn toàn khác nhau. Giai đoạn này nhằm dạy Projector cách chuyển đổi mà không làm hỏng tri thức sẵn có của LLM.
- **Phương pháp:**
  - *Image-Text Contrastive Learning (CLIP InfoNCE):* Huấn luyện cặp vector ảnh và chữ kéo lại gần nhau nếu cùng mô tả một vật, đẩy ra xa nếu khác nhau.
  - *Image-Text Matching (ITM):* Bài toán phân loại nhị phân dự đoán xem đoạn văn bản có khớp chính xác với ảnh không.
  - *Feature Alignment (LLaVA Stage 1):* Đóng băng hoàn toàn Vision Encoder và LLM, chỉ cập nhật ma trận trọng số của Projector trên các cặp ảnh – chú thích ngắn (Image-Caption).

#### b. Giai đoạn 2: Multimodal Instruction Tuning (Visual SFT - Tinh chỉnh theo chỉ dẫn)
- **Ý tưởng:** Sau giai đoạn 1, mô hình mới chỉ biết mô tả ảnh sơ sài. Giai đoạn này biến mô hình thành một trợ lý đa phương thức thực thụ, biết hội thoại tự nhiên, hiểu câu hỏi phức tạp và suy luận logic.
- **Phương pháp:** Mở khóa toàn bộ trọng số của LLM Backbone và Projector. Huấn luyện trên tập dữ liệu hội thoại đa dạng (LLaVA-Instruct, ShareGPT4V) gồm 3 dạng:
  1. *Hội thoại nhiều lượt (Conversation)*
  2. *Mô tả chi tiết sâu (Detailed Description)*
  3. *Suy luận phức tạp nhiều bước (Complex Reasoning)*

#### c. Giai đoạn 3: Multimodal Preference Alignment (Căn chỉnh sở thích đa phương thức)
- **Ý tưởng:** Giảm thiểu hiện tượng ảo giác (bịa đặt thông tin không có trong ảnh) và đảm bảo an toàn.
- **Phương pháp:** Áp dụng **Direct Preference Optimization (DPO)** hoặc **RLHF**. Cung cấp cho mô hình bộ dữ liệu: (Ảnh, Câu hỏi, Câu trả lời đúng thực tế $y_w$, Câu trả lời bịa đặt ảo giác $y_l$) để mô hình tự động tối ưu hóa xác suất nghiêng về câu trả lời trung thực với bức ảnh.

---

### 5.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Sơ đồ mục tiêu, dữ liệu và kết quả từng giai đoạn

| Giai đoạn | Dữ liệu đầu vào | Trọng số được cập nhật | Mục tiêu tối ưu hóa (Loss) | Kết quả đạt được |
|---|---|---|---|---|
| **Giai đoạn 1: Pretraining (Feature Alignment)** | 600K - 1M cặp (Ảnh, Caption ngắn) lọc từ CC3M, LAION. | **Chỉ cập nhật Projector** *(Đóng băng ViT và LLM)* | Dự đoán token caption tiếp theo qua Cross-Entropy: $\mathcal{L} = -\sum \log P(t_i \mid t_{<i}, I)$ | Projector học được cách ánh xạ visual feature sang không gian text token của LLM. |
| **Giai đoạn 2: Visual Instruction Tuning (SFT)** | 150K - 1M mẫu chỉ dẫn chất lượng cao (Hội thoại, phân tích lý luận, giải toán). | **Mở khóa LLM Backbone + Projector** *(ViT đóng băng)* | Cực tiểu hóa Cross-Entropy loss trên toàn bộ chuỗi token câu trả lời của trợ lý. | Mô hình trở thành trợ lý đa phương thức: biết trả lời VQA, phân tích bảng biểu, làm thơ theo ảnh. |
| **Giai đoạn 3: Preference Alignment (DPO/RLHF)** | Hàng chục ngàn bộ ba: (Ảnh, Câu hỏi, Câu trả lời chuẩn $y_w$, Câu trả lời ảo giác $y_l$). | **LLM Backbone** | Tối ưu hàm mất mát DPO: Tăng biên độ xác suất giữa $y_w$ và $y_l$ mà không làm lệch xa mô hình SFT. | Giảm tỷ lệ ảo giác vật thể (Object Hallucination), tăng tính an toàn và trung thực. |

---

## 6. Quá trình sinh phản hồi

### 6.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

#### a. Cơ chế tiếp nhận và sinh tự hồi quy (Autoregressive Generation)
VLM sinh văn bản dựa trên nguyên lý dự đoán token tiếp theo có điều kiện tiền tố (Conditional Next-Token Prediction):
1. **Tiền tố thị giác (Visual Prefix):** Chuỗi vector ảnh $[H_v^{(1)}, \dots, H_v^{(N_v)}]$ được đưa vào đầu tiên như một "ngữ cảnh bất biến".
2. **Kích hoạt KV Cache:** Mô hình tính toán sẵn Key và Value của các visual tokens để tái sử dụng trong các bước giải mã, giúp tiết kiệm thời gian tính toán.
3. **Sinh tuần tự:** Tại mỗi bước thời gian $t$, mô hình tính xác suất cho token tiếp theo dựa trên toàn bộ ảnh và các từ đã sinh trước đó:
   $$P(w_t \mid I, w_1, w_2, \dots, w_{t-1}) = \text{Softmax}\left(\frac{z_t}{T}\right)$$

#### b. Vai trò của các chiến lược giải mã (Decoding Strategies)
- **Greedy Decoding ($T = 0$):** Luôn chọn token có xác suất cao nhất. Phù hợp cho các tác vụ cần độ chính xác tuyệt đối như OCR đọc số hóa đơn, nhận diện biển số xe, giải toán hình học.
- **Temperature ($T > 0$):** Điều chỉnh độ mượt của phân phối xác suất. $T$ thấp ($0.2 - 0.5$) giúp câu trả lời tập trung, bám sát ảnh; $T$ cao ($0.7 - 1.0$) kích hoạt tính sáng tạo khi yêu cầu mô hình làm thơ hoặc viết câu chuyện dựa trên ảnh.
- **Top-$p$ (Nucleus Sampling):** Giới hạn tập token ứng viên trong nhóm có tổng xác suất tích lũy đạt ngưỡng $p$ (ví dụ $0.9$), loại bỏ các từ ngữ vô nghĩa ở đuôi phân phối.

---

### 6.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Cùng một hình ảnh với các prompt khác nhau

**Hình ảnh thử nghiệm:** Bức ảnh chụp một ngã tư đường phố giờ cao điểm tại TP.HCM: trời bắt đầu mưa lấm tấm, xe máy ken đặc dừng trước vạch đỏ, một chiếc xe buýt lớn màu xanh đang chuẩn bị rẽ phải, và có một người mặc áo mưa tiện lợi màu vàng đang đi bộ sát mép vỉa hè.

```
+---------------------------------------------------------------------------------------+
| HÌNH ẢNH MINH HỌA: Ngã tư giao thông giờ cao điểm trời mưa                           |
+---------------------------------------------------------------------------------------+
| PROMPT 1 (Mô tả tổng quan):                                                           |
| "Hãy mô tả ngắn gọn khung cảnh trong bức ảnh này trong 2 câu."                        |
|                                                                                       |
| PHẢN HỒI CỦA VLM:                                                                     |
| "Bức ảnh ghi lại cảnh ngã tư đường phố đông đúc vào giờ cao điểm dưới trời mưa nhẹ.   |
| Dòng xe máy đang dừng chờ đèn đỏ bên cạnh một chiếc xe buýt lớn chuẩn bị rẽ."          |
+---------------------------------------------------------------------------------------+
| PROMPT 2 (Phát hiện nguy cơ an toàn giao thông):                                      |
| "Bức ảnh này có yếu tố nào tiềm ẩn nguy cơ mất an toàn giao thông không? Giải thích."|
|                                                                                       |
| PHẢN HỒI CỦA VLM:                                                                     |
| "Có hai yếu tố nguy cơ đáng chú ý:                                                    |
| 1. Chiếc xe buýt lớn đang rẽ phải có thể tạo ra điểm mù lớn đối với các xe máy đi sát.|
| 2. Người đi bộ mặc áo mưa vàng đi dưới lòng đường sát vạch rẽ trong điều kiện đường  |
|    trơn ướt và tầm nhìn hạn chế, rất dễ va chạm nếu tài xế xe buýt thiếu quan sát."   |
+---------------------------------------------------------------------------------------+
| PROMPT 3 (Trích xuất văn bản & Biển hiệu - OCR):                                       |
| "Đọc tuyến xe buýt và các biển báo có thể nhìn thấy trong hình."                     |
|                                                                                       |
| PHẢN HỒI CỦA VLM:                                                                     |
| "- Xe buýt mang bảng số tuyến: 'Tuyến số 08: Bến xe Quận 8 - ĐH Quốc Gia'.           |
|  - Biển báo giao thông treo phía trên góc phải: 'Cấm quay đầu xe'."                  |
+---------------------------------------------------------------------------------------+
| PROMPT 4 (Sáng tạo nội dung):                                                         |
| "Viết một đoạn nhật ký ngắn (khoảng 3 câu) dưới góc nhìn của người đi xe máy trong ảnh."|
|                                                                                       |
| PHẢN HỒI CỦA VLM:                                                                     |
| "Tiếng còi xe inh ỏi hòa cùng những giọt mưa đầu mùa bắt đầu táp vào mặt rát buốt.   |
| Tôi nhìn chiếc xe buýt đồ sộ nhích từng centimet bên cạnh mà lòng chỉ mong đèn sớm    |
| chuyển sang xanh để kịp về nhà ăn bữa cơm tối cùng gia đình."                        |
+---------------------------------------------------------------------------------------+
```

---

## 7. Khả năng của VLM

### 7.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

Năng lực của VLM hiện đại bao phủ 5 miền tác vụ trọng tâm:
1. **Visual Question Answering (VQA):** Trả lời các câu hỏi tự do bằng ngôn ngữ tự nhiên dựa trên thông tin thị giác trong ảnh (từ mức độ nhận biết thực thể đến hiểu bối cảnh văn hóa, cảm xúc).
2. **Image Captioning & Rich Dense Captioning:** Mô tả nội dung bức ảnh từ câu tóm tắt đơn giản đến bản báo cáo chi tiết từng milimet không gian.
3. **OCR-Free Document Understanding:** Đọc hiểu trực tiếp văn bản trong tự nhiên (biển hiệu, áo thun) và tài liệu phức tạp (hóa đơn, bảng cân đối kế toán, biểu mẫu, sơ đồ mạng) mà không cần qua module OCR truyền thống.
4. **Visual Reasoning & Math Solving:** Giải toán hình học, đọc đồ thị hàm số, suy luận logic từ các sơ đồ kỹ thuật (Multimodal Chain-of-Thought).
5. **Visual Grounding & Object Referencing:** Trỏ chính xác vị trí vật thể bằng bounding box `[ymin, xmin, ymax, xmax]` tương ứng với tên gọi trong văn bản.

---

### 7.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Thử nghiệm nhiều tác vụ chuyên sâu

Bảng tổng hợp thử nghiệm đánh giá đa năng lực của mô hình VLM trên các loại hình ảnh khác nhau:

| Tác vụ thử nghiệm | Dạng hình ảnh | Prompt yêu cầu | Kết quả thực thi của VLM | Đánh giá năng lực |
|---|---|---|---|---|
| **VQA Y khoa** | Phim chụp X-quang lồng ngực | *"Có tổn thương đông đặc hoặc tràn dịch màng phổi ở bên nào không?"* | *"Phát hiện vùng mờ không đồng nhất ở thùy dưới phổi phải, góc sườn hoành phải bị tù, gợi ý dấu hiệu tràn dịch màng phổi lượng ít."* | **Rất tốt:** Nhận biết được cấu trúc giải phẫu và biến đổi đậm độ mô học cơ bản. |
| **DocVQA / Hóa đơn** | Hóa đơn tiền điện dạng ảnh chụp nghiêng | *"Chỉ số công tơ tháng này là bao nhiêu và lượng điện tiêu thụ là mấy kWh?"* | *"Chỉ số mới: 4520, chỉ số cũ: 4210. Sản lượng điện năng tiêu thụ trong kỳ là 310 kWh."* | **Xuất sắc:** Tự động nắn chỉnh góc nghiêng và đọc bảng số liệu không cần OCR phụ trợ. |
| **Visual Math** | Bài toán hình học có hình vẽ tam giác vuông | *"Tính chiều dài cạnh BC trong hình vẽ tam giác vuông ABC tại A, biết AB = 6cm, góc B = 60 độ."* | *"Áp dụng công thức lượng giác trong tam giác vuông: $\cos(B) = \frac{AB}{BC} \implies BC = \frac{AB}{\cos(60^\circ)} = \frac{6}{0.5} = 12\text{ cm}$."* | **Xuất sắc:** Kết hợp nhìn hình trích xuất dữ kiện và suy luận công thức toán học logic. |
| **ChartQA** | Biểu đồ tròn cơ cấu doanh thu | *"Mảng dịch vụ đám mây chiếm bao nhiêu phần trăm và đứng thứ mấy trong cơ cấu?"* | *"Mảng dịch vụ đám mây chiếm 34.5% tổng doanh thu và là mảng đóng góp lớn nhất (đứng vị trí thứ nhất)."* | **Rất tốt:** Phân biệt được màu sắc các lát cắt biểu đồ tương ứng với bảng chú giải (legend). |

---

## 8. Cách thích nghi và ứng dụng VLM

### 8.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

Để đưa một mô hình VLM nền tảng (Foundation Model) vào ứng dụng chuyên biệt của doanh nghiệp hoặc nghiên cứu chuyên sâu, có 4 phương pháp chính:

#### a. Multimodal Prompting & In-Context Learning (ICL)
- Không can thiệp vào trọng số mô hình.
- Cung cấp hướng dẫn rõ ràng trong System Prompt kết hợp đưa vài cặp ảnh ví dụ mẫu (Few-shot) vào ngữ cảnh để mô hình học theo phong cách trả lời.
- Chi phí bằng 0 về mặt huấn luyện, triển khai ngay lập tức.

#### b. Full Fine-tuning
- Cập nhật toàn bộ tham số của cả Projector và LLM Backbone trên tập dữ liệu chuyên ngành (ví dụ dữ liệu ảnh vi mạch, hồ sơ bệnh án).
- Đạt hiệu năng tối đa trên miền dữ liệu hẹp nhưng chi phí phần cứng cực kỳ đắt đỏ và đối mặt với nguy cơ "quên kiến thức cũ" (Catastrophic Forgetting).

#### c. Parameter-Efficient Fine-Tuning (PEFT / LoRA)
- Đóng băng toàn bộ trọng số gốc của mô hình. Chỉ gắn thêm các nhánh ma trận hạng thấp (Low-Rank Adaptation - LoRA) vào các tầng Attention của LLM Backbone hoặc Projector.
- Giảm hơn 90% bộ nhớ GPU cần thiết khi huấn luyện; có thể dễ dàng chuyển đổi qua lại giữa nhiều tác vụ bằng cách hoán đổi các file adapter trọng số rất nhẹ (vài chục MB).

#### d. Multimodal RAG (Retrieval-Augmented Generation)
- Kết hợp VLM với kho cơ sở dữ liệu vector đa phương thức.
- Khi người dùng hỏi kèm ảnh, hệ thống dùng mô hình embedding (như CLIP hoặc ColPali) để truy xuất các tài liệu, sơ đồ, hướng dẫn sử dụng liên quan nhất trong kho lưu trữ, rồi ghép vào prompt để VLM trả lời có dẫn chứng tin cậy.

---

### 8.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Bảng so sánh 4 phương pháp thích nghi VLM

| Phương pháp | Dữ liệu cần có | Chi phí tính toán / GPU | Ưu điểm nổi bật | Nhược điểm / Hạn chế | Trường hợp sử dụng thực tế |
|---|---|---|---|---|---|
| **Prompting & In-Context Learning** | 1 - 5 ảnh mẫu kèm câu trả lời mẫu | **Không tốn chi phí train** (chỉ tốn phí inference) | Nhanh, thử nghiệm được ngay, không cần hạ tầng GPU để train | Giới hạn bởi độ dài ngữ cảnh (Context Window); không nạp thêm được lượng tri thức lớn | Thử nghiệm nhanh ý tưởng, trích xuất thông tin theo mẫu form cố định |
| **PEFT / LoRA (Adapter)** | 1.000 - 50.000 cặp (Ảnh, Lời giải chuẩn) | **Thấp - Trung bình** (1-2 GPU dân dụng như RTX 3090/4090) | Tiết kiệm 80-90% VRAM, bảo toàn tri thức nền của LLM, file trọng số gọn nhẹ | Cần chuẩn bị dữ liệu gán nhãn chuẩn; không cập nhật tri thức thời gian thực | Tùy biến VLM đọc bản vẽ CAD cơ khí, hóa đơn đặc thù ngành, thời trang |
| **Full Fine-Tuning** | > 100.000 dữ liệu đa phương thức quy mô lớn | **Rất cao** (Cụm nhiều máy chủ A100/H100) | Tối ưu hóa sâu toàn diện năng lực mô hình trên toàn miền dữ liệu | Chi phí huấn luyện khổng lồ; dễ gặp hiện tượng quên kiến thức tổng quát | Xây dựng các mô hình nền tảng y tế quốc gia, mô hình VLM quân sự chuyên dụng |
| **Multimodal RAG** | Kho tài liệu PDF, cẩm nang kỹ thuật, hình ảnh sản phẩm | **Trung bình** (Chi phí xây dựng vector database và embedding) | Cập nhật tri thức mới tức thì, giảm triệt để ảo giác, trích dẫn được trang tài liệu nguồn | Phụ thuộc vào chất lượng module truy xuất (Retriever); độ trễ phản hồi cao hơn | Trợ lý tra cứu hướng dẫn bảo trì máy bay, hỏi đáp chính sách trên hợp đồng bảo hiểm |

---

## 9. Hạn chế và đánh giá

### 9.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

#### a. Các hạn chế cố hữu của VLM hiện đại
1. **Object Hallucination (Ảo giác thị giác):** Mô hình khẳng định chắc chắn có một vật thể hoặc hành động trong ảnh dù thực tế không hề có.
   - *Nguyên nhân:* Do thiên lệch ngôn ngữ (Language Prior) của LLM Backbone lấn át tín hiệu thị giác. Nếu trong văn bản từ "bếp" thường đi kèm "lò vi sóng", thì khi thấy ảnh căn bếp trống, mô hình vẫn tự bịa ra "lò vi sóng".
2. **Hạn chế độ phân giải và nhận diện chi tiết nhỏ:** Do ảnh bị nén thành các patch $14 \times 14$ pixel, các vật thể chỉ chiếm vài pixel (biển số xe từ xa, vết xước nhỏ) dễ bị mờ nhạt hoặc biến mất sau lớp Vision Encoder.
3. **Kém về suy luận không gian 3D và đếm chính xác (Counting):** VLM rất hay đếm sai khi số lượng vật thể vượt quá 5 (ví dụ đếm số quả táo trong giỏ) và dễ nhầm lẫn các quan hệ không gian ("nằm bên trái" vs "nằm bên phải", "phía trước" vs "phía sau").
4. **Chi phí tính toán tăng vọt theo độ dài chuỗi (Token Explosion):** Một bức ảnh có thể ngốn từ 576 đến hàng ngàn tokens, chiếm dụng bộ nhớ KV Cache và làm chậm tốc độ suy luận gấp nhiều lần so với văn bản thuần túy.

#### b. Các phương pháp đánh giá chất lượng VLM
- **POPE (Polling-based Object Probing Evaluation):** Đo lường mức độ ảo giác vật thể thông qua tập câu hỏi Yes/No với các bẫy đối kháng (Adversarial).
- **MME & MMBench:** Đánh giá toàn diện 14-20 khía cạnh từ mức độ nhận thức (Perception) đến nhận thức bậc cao (Cognition, Reasoning).
- **MathVista:** Đánh giá khả năng giải toán và tư duy logic thị giác.
- **DocVQA / InfoVQA:** Đánh giá khả năng đọc hiểu tài liệu thực tế.

---

### 9.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Ví dụ mô hình mô tả sai hình ảnh và cách kiểm chứng

#### a. Trường hợp thực nghiệm xảy ra ảo giác thực tế
- **Hình ảnh thử nghiệm:** Bức ảnh chụp cận cảnh ghi-đông một chiếc xe đạp thể thao cũ dựng cạnh hàng rào. Trên ghi-đông chỉ có tay phanh và một chiếc đồng hồ đo tốc độ điện tử nhỏ màu đen, **hoàn toàn không có chuông xe đạp**.
- **Prompt:** *"Chiếc xe đạp này có chuông không? Nếu có thì nó nằm ở vị trí nào và có màu gì?"*

```
+---------------------------------------------------------------------------------------+
| PHẢN HỒI SAI LỆCH CỦA MÔ HÌNH (HALLUCINATION):                                        |
| "Có, chiếc xe đạp này có một chiếc chuông kim loại nhỏ màu bạc được gắn ở phía bên    |
| trái của tay lái, ngay cạnh tay phanh."                                               |
+---------------------------------------------------------------------------------------+
| PHÂN TÍCH LỖI:                                                                        |
| 1. Lỗi thực tế: Bức ảnh hoàn toàn không có chuông xe đạp.                             |
| 2. Bản chất ảo giác: Mô hình bị ảnh hưởng bởi thiên lệch ngôn ngữ mạnh mẽ: trong dữ   |
|    liệu văn bản, các từ "xe đạp", "tay lái", "tay phanh" hầu như luôn xuất hiện cùng |
|    cụm từ "chuông xe đạp màu bạc". LLM đã tự động "điền vào chỗ trống" để câu văn    |
|    nghe trôi chảy nhất mà bỏ qua tín hiệu thực tế từ Vision Encoder.                  |
+---------------------------------------------------------------------------------------+
```

#### b. Quy trình kiểm chứng và khắc phục lỗi ảo giác
1. **Kiểm chứng bằng Visual Grounding:** Yêu cầu mô hình xuất ra tọa độ Bounding Box của vật thể nghi vấn: *"Hãy trỏ tọa độ của chiếc chuông đó `[ymin, xmin, ymax, xmax]`"*. Khi buộc phải xuất tọa độ, mô hình sẽ lộ rõ sự lúng túng (trỏ vào khoảng trống hoặc trỏ nhầm sang chiếc đồng hồ đo tốc độ).
2. **Kỹ thuật Prompting phản trực giác (Contrastive / Verification Prompting):** Thêm ràng buộc: *"Chỉ trả lời dựa trên những gì mắt bạn nhìn thấy rõ 100%. Nếu không nhìn thấy chuông, hãy dứt khoát trả lời 'Không có'. Không được tự suy diễn."*
3. **Áp dụng POPE Pipeline:** Đo lường tỷ lệ trả lời đúng trên 3 tập câu hỏi Yes/No: Random, Popular và Adversarial để tính chỉ số F1-Score thực tế của mô hình.

---

## 10. Demo và kết luận

### 10.1 Cơ sở lý thuyết & Nguyên lý cốt lõi

#### a. Thiết kế thực nghiệm
Thực nghiệm đánh giá năng lực của mô hình VLM nguồn mở (ví dụ: Qwen2-VL-7B-Instruct hoặc LLaVA-1.5-7B) trên bộ 20 mẫu thử nghiệm đa dạng thuộc 4 nhóm tác vụ:
1. Nhận diện và mô tả vật thể thông thường (Object Recognition / Captioning - 5 mẫu)
2. Đọc hiểu tài liệu thực tế và hóa đơn (DocVQA - 5 mẫu)
3. Suy luận toán học & biểu đồ (Chart & Geometry Reasoning - 5 mẫu)
4. Tác vụ bẫy ảo giác đối kháng (Hallucination Probing - 5 mẫu)

#### b. Tổng hợp kết quả và phân tích ma trận lỗi
- **Nhóm tác vụ đạt độ chính xác cao nhất:** Mô tả ảnh tổng quan (100%) và Đọc hiểu tài liệu rõ nét (90%).
- **Nhóm tác vụ gặp nhiều lỗi nhất:** Đếm số lượng vật thể nhỏ (> 6 đối tượng) và suy luận hình học phức tạp.
- **Phân loại lỗi chính ghi nhận được:**
  - *Lỗi độ phân giải (Resolution Loss):* Chữ quá nhỏ trong hóa đơn bị đọc nhầm ký tự (nhầm số 8 thành số 3 hoặc 0).
  - *Lỗi ảo giác (Hallucination):* Tự thêm các chi tiết quen thuộc trong đời sống vào ảnh.
  - *Lỗi suy luận logic đa bước:* Đọc đúng dữ kiện từ biểu đồ nhưng tính toán cộng trừ sai ở bước cuối.

---

### 10.2 Minh họa kỹ thuật & Tác vụ thực nghiệm: Bảng kết quả thực nghiệm và phân tích lỗi chi tiết

#### a. Bảng tổng hợp số liệu thực nghiệm

| Nhóm tác vụ | Số mẫu thử nghiệm | Số mẫu đúng hoàn toàn | Tỷ lệ chính xác (%) | Dạng lỗi phổ biến nhất |
|---|---|---|---|---|
| **Mô tả ảnh & VQA cơ bản** | 5 | 5 / 5 | **100.0%** | Không có lỗi nghiêm trọng, câu văn lưu loát. |
| **Đọc hiểu tài liệu & Hóa đơn** | 5 | 4 / 5 | **80.0%** | Sai sót ở ký tự số bị mờ hoặc viết tắt chuyên ngành. |
| **Biểu đồ & Toán hình học** | 5 | 3 / 5 | **60.0%** | Nhầm lẫn tỷ lệ cột biểu đồ; sai số tính toán đa bước. |
| **Kiểm tra bẫy ảo giác (POPE)** | 5 | 3 / 5 | **60.0%** | Bị lừa ở các câu hỏi bẫy về vật thể phổ biến (xe đạp, đồng hồ). |
| **TỔNG HỢP TOÀN BỘ** | **20** | **15 / 20** | **75.0%** | Điểm số phản ánh đúng năng lực thực tế của VLM 7B. |

#### b. Minh họa chi tiết 3 ca thử nghiệm điển hình

```
=========================================================================================
CA THỰC NGHIỆM 1: ĐỌC TÀI LIỆU HÓA ĐƠN (ĐẠT KẾT QUẢ XUẤT SẮC)
=========================================================================================
- ẢNH ĐẦU VÀO: Hóa đơn tiền cà phê chụp bằng điện thoại.
- CÂU HỎI: "Món đồ uống nào đắt nhất và đơn giá là bao nhiêu?"
- TRẢ LỜI CỦA VLM: "Món đồ uống đắt nhất trong hóa đơn là 'Trà đào cam sả' với đơn giá 55.000 VNĐ."
- NHÃN ĐÚNG (GROUND TRUTH): Trà đào cam sả - 55.000 VNĐ.
- ĐÁNH GIÁ: ĐÚNG TUYỆT ĐỐI. Mô hình định vị chính xác cột Tên món và cột Thành tiền.

=========================================================================================
CA THỰC NGHIỆM 2: SUY LUẬN BIỂU ĐỒ (LỖI TÍNH TOÁN ĐA BƯỚC)
=========================================================================================
- ẢNH ĐẦU VÀO: Biểu đồ cột biểu diễn sản lượng xuất khẩu gạo qua các quý 1, 2, 3, 4.
- CÂU HỎI: "Tổng sản lượng gạo xuất khẩu của Quý 1 và Quý 2 chênh lệch bao nhiêu tấn so với Quý 3?"
- TRẢ LỜI CỦA VLM: "Sản lượng Quý 1 là 1.2 triệu tấn, Quý 2 là 1.5 triệu tấn (tổng 2.7 triệu tấn).
                   Quý 3 là 1.8 triệu tấn. Độ chênh lệch là: 2.7 - 1.8 = 1.1 triệu tấn."
- NHÃN ĐÚNG: 2.7 - 1.8 = 0.9 triệu tấn.
- ĐÁNH GIÁ: SAI Ở BƯỚC TÍNH TOÁN CUỐI. Mô hình đọc đúng 100% dữ liệu thị giác từ biểu đồ nhưng
            bộ não LLM thực hiện phép trừ số học sai (ra 1.1 thay vì 0.9).

=========================================================================================
CA THỰC NGHIỆM 3: BẪY ẢO GIÁC VẬT THỂ (LỖI THIÊN LỆCH NGÔN NGỮ)
=========================================================================================
- ẢNH ĐẦU VÀO: Bàn ăn chỉ có đĩa bít tết và ly rượu vang đỏ, hoàn toàn không có dao nĩa.
- CÂU HỎI: "Có chiếc dao cắt thịt nào đặt cạnh đĩa bít tết không?"
- TRẢ LỜI CỦA VLM: "Có, có một chiếc dao cắt thịt bằng kim loại đặt ở phía bên phải đĩa bít tết."
- NHÃN ĐÚNG: Không có dao.
- ĐÁNH GIÁ: SAI DO ẢO GIÁC (HALLUCINATION). LLM tự suy diễn theo thói quen ẩm thực phương Tây.
=========================================================================================
```

### 10.3 Kết luận & Đánh giá Kỹ thuật LVM / VLM
1. **LVM / VLM là cột mốc tiến hóa quan trọng:** Biến trí tuệ nhân tạo từ việc chỉ thao tác trên con chữ trừu tượng thành hệ thống có thể nhìn, thấu hiểu và tương tác với thế giới thực.
2. **Sự phối hợp hoàn hảo của 3 module:** Thành công của mô hình không đòi hỏi phải thiết kế lại từ đầu, mà là nghệ thuật kết hợp giữa một *Vision Encoder* mạnh mẽ, một *Projector* tinh gọn và một *LLM Backbone* giàu tri thức suy luận.
3. **Thách thức cốt lõi:** Để ứng dụng vào các lĩnh vực độ chính xác cao (Y tế, Hàng không, Xe tự hành), các hệ thống LVM cần giải quyết dứt điểm bài toán **Ảo giác thị giác (Hallucination)** và tối ưu hóa **độ dài ngữ cảnh visual token**.
4. **Xu hướng tương lai tất yếu:** Chuyển dịch từ việc "ghép nối module rời rạc" sang **Mô hình Đa phương thức Nguyên bản (Native Multimodal như GPT-4o, Gemini)** và bước chân vào thế giới thực dưới dạng **Trí tuệ thể hiện (Embodied AI / Robotics)**.

---

## Tài liệu tham khảo học thuật

1. **Vaswani et al. (2017)** — *Attention Is All You Need*. NIPS 2017.
2. **Dosovitskiy et al. (2020)** — *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)*. ICLR 2021.
3. **Radford et al. (2021)** — *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML 2021.
4. **Alayrac et al. (2022)** — *Flamingo: a Visual Language Model for Few-Shot Learning*. NeurIPS 2022.
5. **Li et al. (2023)** — *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML 2023.
6. **Liu et al. (2023)** — *Visual Instruction Tuning (LLaVA)*. NeurIPS 2023.
7. **Bai et al. (2023)** — *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond*. arXiv:2308.12966.
8. **Wang et al. (2024)** — *Qwen2-VL: To See the World More Clearly*. arXiv:2409.12191.
9. **Li et al. (2023)** — *Evaluating Object Hallucination in Large Vision-Language Models (POPE)*. EMNLP 2023.
10. **Lu et al. (2023)** — *MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts*. ICLR 2024.
