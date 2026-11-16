# Demo: Tìm hiểu mô hình ngôn ngữ lớn (LLMs)

Gói thực nghiệm phục vụ tìm hiểu các nguyên lý cốt lõi của Mô hình Ngôn ngữ Lớn (Large Language Models — LLMs), bao gồm kiến trúc Transformer decoder-only, cơ chế causal attention, quá trình tokenization, nguyên lý học next-token prediction, các chiến lược giải mã văn bản, năng lực in-context learning (zero-shot và few-shot), mô hình truy xuất tăng cường (RAG) và các hạn chế cố hữu (ảo giác, giới hạn ngữ cảnh).

## 1. Cấu trúc thư mục

```
llm_demo.ipynb            Notebook thực nghiệm và trực quan hóa kết quả
train_minigpt.py          Script huấn luyện mô hình MiniGPT mức ký tự
run_llm_experiment.py     Script thực nghiệm zero-shot và few-shot trên LLM
src/
  minigpt.py              Kiến trúc GPT decoder-only và các phương pháp giải mã
  tok_utils.py            Bộ công cụ nạp và xử lý tokenizer (GPT-2, Llama 2)
  gpt2_bpe_openai.py      Bộ giải mã BPE byte-level
data/                     Ngữ liệu văn bản, từ vựng tokenizer, bộ test cảm xúc
checkpoints/minigpt.pt    Trọng số mô hình MiniGPT đã huấn luyện
results/                  Dữ liệu kết quả thực nghiệm (CSV, JSON)
figures/                  Biểu đồ và sơ đồ trực quan hóa
```

## 2. Cài đặt và môi trường

Yêu cầu môi trường Python 3.8 trở lên:

```bash
pip install -r requirements.txt
```

Để thực thi toàn bộ notebook và tái hiện kết quả:
```bash
jupyter nbconvert --to notebook --execute llm_demo.ipynb --output llm_demo.ipynb
```
*Lưu ý:* Notebook nạp sẵn trọng số từ `checkpoints/minigpt.pt`. Nếu cần huấn luyện lại mô hình MiniGPT từ đầu:
```bash
python train_minigpt.py
```

## 3. Các thành phần thực nghiệm

Dự án phân định rõ hai thành phần thực nghiệm phục vụ hai mục tiêu phân tích khác nhau:

| Thành phần | Đặc điểm kỹ thuật | Mục tiêu khảo sát |
|---|---|---|
| **Tokenizer thật (GPT-2, Llama 2)** | BPE byte-level (từ vựng 50.257) và SentencePiece BPE (từ vựng 32.000) | Đánh giá hiệu suất phân đoạn subword, biểu diễn token ID và độ giãn nở ngữ cảnh tiếng Việt vs tiếng Anh (Mục 3, 9) |
| **Mô hình MiniGPT** | Transformer decoder-only (~0,8M tham số), mức ký tự, huấn luyện trên tinyshakespeare | Khảo sát trực tiếp cơ chế causal attention, hàm mất mát cross-entropy và ảnh hưởng của các tham số giải mã ($T$, top-$p$, greedy) (Mục 2, 4, 6, 9) |
| **Mô hình LLM (Qwen 2.5 7B)** | Mô hình chỉ dẫn quy mô lớn, chạy suy luận có kiểm soát | Khảo sát năng lực in-context learning qua so sánh zero-shot vs few-shot và hiện tượng ảo giác (Mục 7, 9, 10) |

## 4. Tổng hợp kết quả thực nghiệm chính

| Nội dung | Kết quả thực nghiệm ghi nhận |
|---|---|
| **Causal Attention (Mục 2)** | Trọng số attention ở các vị trí tương lai ($> t$) triệt tiêu về 0 trên toàn bộ các head và layer. Biến đổi dữ liệu sau vị trí $t=20$ không gây sai lệch (chênh lệch logits = 0,0) lên các vị trí trước đó. |
| **Tokenization (Mục 3)** | Trên 7 cặp câu song ngữ tương đương, tỷ lệ token tiếng Việt/tiếng Anh trung bình là **3,92 lần** (với GPT-2 BPE) và **2,65 lần** (với Llama 2 SentencePiece) do hiện tượng tách byte đơn lẻ ở các ký tự có dấu. |
| **Học tự giám sát (Mục 4)** | Quá trình huấn luyện MiniGPT hội tụ sau 2.500 bước với loss kiểm định đạt 1,579 nats/ký tự (so với khởi tạo ngẫu nhiên $\ln 65 \approx 4,174$). Loss từng vị trí khớp chính xác với hàm cross-entropy lý thuyết. |
| **Chiến lược giải mã (Mục 6)** | Khảo sát 6 cấu hình sinh văn bản (30 seed mỗi cấu hình): Greedy decoding sinh mẫu xác định nhưng lặp vòng 40,5% bộ ba từ; nhiệt độ $T$ tăng làm tăng độ đa dạng từ vựng nhưng tăng tỷ lệ từ lạ (từ 1,7% ở $T=0,3$ lên 38,2% ở $T=1,5$); cấu hình kết hợp $T=0,8$, top-$p=0,95$ cân bằng tốt nhất giữa tính phong phú và tính hợp lệ cú pháp. |
| **In-context Learning (Mục 7 & 10)** | Trên tập kiểm tra 20 câu tiếng Việt, Few-shot (6 ví dụ) đạt độ chính xác **85,0%** (17/20), cải thiện so với Zero-shot (**80,0%**, 16/20) nhờ nhận diện đúng câu mỉa mai ngầm; cả hai chế độ đều đạt độ chính xác 100% trên tập câu dễ (13/13) và 0% lỗi định dạng. |
| **Giới hạn và Ảo giác (Mục 9)** | Thông tin nằm ngoài cửa sổ ngữ cảnh (128 ký tự trên MiniGPT) không gây ảnh hưởng tới phân phối đầu ra (chênh lệch logits = 0,0). Thử nghiệm truy vấn cấp độ ký tự trên LLM bộc lộ hiện tượng ảo giác do rào cản phân tách subword tokenization. |

## 5. Hướng dẫn thực thi thử nghiệm LLM

Thử nghiệm in-context learning được quản lý qua script `run_llm_experiment.py`:

```bash
# 1. Chạy với server cục bộ tương thích OpenAI (Ollama, vLLM, LM Studio):
python run_llm_experiment.py --backend openai_compat --model qwen2.5:7b --base-url http://localhost:11434/v1

# 2. Chạy với Anthropic API:
export ANTHROPIC_API_KEY="your-api-key"
python run_llm_experiment.py --backend anthropic --model claude-3-5-sonnet-20241022

# 3. Chế độ nhập thủ công (không dùng API):
python run_llm_experiment.py --backend manual --export
# Điền kết quả phản hồi vào cột 'output' trong results/manual_prompts.csv, sau đó chấm điểm:
python run_llm_experiment.py --backend manual --score results/manual_prompts.csv --model "<tên_mô_hình>"
```

Các tệp kết quả đầu ra được tự động cập nhật vào thư mục `results/` bao gồm: `llm_predictions.csv`, `llm_summary.csv`, `llm_errors.csv` và `llm_meta.json`.

## 6. Nguồn dữ liệu

- `tinyshakespeare.txt`: Kho ngữ liệu kịch Shakespeare trích xuất từ kho lưu trữ công khai.
- Tokenizer weights: `vocab.bpe` và `encoder.json` từ OpenAI GPT-2; `tokenizer.model` từ Meta Llama 2.
- Ngữ liệu phân loại cảm xúc: Bộ 20 câu tiếng Việt kèm 6 ví dụ few-shot được biên soạn có kiểm soát theo các quy tắc gán nhãn tại `data/labeling_rules.md`.
- Ngữ liệu RAG mô phỏng: Kho văn bản thông tin học phần CS417 phục vụ thử nghiệm trích xuất tài liệu dựa trên độ đo tương đồng TF-IDF.
