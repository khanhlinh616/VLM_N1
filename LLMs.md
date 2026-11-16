# Tìm hiểu mô hình ngôn ngữ lớn (LLMs)


## Dàn ý tổng thể

| # | Mục | Ý chính cần chốt | Minh họa |
| --- | --- | --- | --- |
| 1 | Giới thiệu | Language model → LLM → chatbot là ba tầng khác nhau | Bảng 4 tác vụ |
| 2 | Kiến trúc | Decoder-only + causal attention | Sơ đồ token → xác suất |
| 3 | Tokenization | Subword, token ID, embedding | Tokenize 1 câu Việt, 1 câu Anh |
| 4 | Nguyên lý học | Next-token prediction, cross-entropy | Cặp ngữ cảnh → token đích |
| 5 | Giai đoạn huấn luyện | Pretraining → SFT → alignment (RLHF/DPO) | Sơ đồ mục tiêu và dữ liệu |
| 6 | Sinh văn bản | Autoregressive, greedy, temperature, top-p | Cùng prompt, nhiều cấu hình |
| 7 | Khả năng | In-context learning, làm theo chỉ dẫn | Zero-shot vs few-shot |
| 8 | Thích nghi | Prompting, fine-tuning, RAG | Bảng so sánh |
| 9 | Hạn chế | Hallucination, thiên lệch, giới hạn ngữ cảnh | 1 câu sai + cách kiểm chứng |
| 10 | Demo, kết luận | Thử nghiệm có kiểm soát | Bảng kết quả, phân tích lỗi |

> [!NOTE]
> Toàn bộ công thức toán học và ký hiệu trong dự án đã được chuẩn hóa và cung cấp mã nguồn theo định dạng **OMML (Office Math Markup Language)** theo chuẩn ISO/IEC 29500-1 dùng cho Microsoft Word, PowerPoint và OpenXML tại tệp [OMML_formulas.md](file:///c:/Project/presen/files/llm_demo/llm_demo/OMML_formulas.md) và [OMML_formulas.xml](file:///c:/Project/presen/files/llm_demo/llm_demo/OMML_formulas.xml).

## 1. Giới thiệu

### Dàn ý

- 1.1 Language model là gì
- 1.2 LLM là gì, chữ “large” nghĩa là gì
- 1.3 Phân biệt language model, LLM và chatbot
- 1.4 Các tác vụ minh họa

### Nội dung

**Language model (mô hình ngôn ngữ)** là mô hình gán xác suất cho một chuỗi token. Theo quy tắc xác suất chuỗi:

$$P(w_1, \dots, w_n) = P(w_1) \prod_{i=2}^n P(w_i \mid w_1, \dots, w_{i-1}) = P(w_1) \times P(w_2 \mid w_1) \times \dots \times P(w_n \mid w_1, \dots, w_{n-1})$$

Nói đơn giản: mô hình trả lời câu hỏi “với đoạn văn đã có, token tiếp theo nhiều khả năng là gì?”. Mô hình ngôn ngữ đã tồn tại từ lâu (n-gram, RNN/LSTM); điểm khác biệt hiện nay là kiến trúc và quy mô. Cách giải thích token tiếp theo ở trên đúng với mô hình ngôn ngữ tự hồi quy (autoregressive), loại được dùng trong tài liệu này; còn có loại khác, ví dụ BERT dùng masked language modeling, tức dự đoán token bị che.

**LLM (Large Language Model)** là mô hình ngôn ngữ quy mô lớn, được huấn luyện trên lượng dữ liệu văn bản lớn với năng lực tính toán đáng kể. Phần lớn LLM hiện đại dùng kiến trúc Transformer, nhưng Transformer hay một ngưỡng số tham số cố định không phải là điều kiện định nghĩa duy nhất. “Large” thể hiện ở ba mặt: số tham số, lượng dữ liệu và lượng tính toán. Ở quy mô đó, mô hình bộc lộ các khả năng tổng quát như làm theo chỉ dẫn, học từ ví dụ trong prompt, viết mã.

**Chatbot** là một *ứng dụng* xây dựng quanh LLM chứ không phải bản thân mô hình.

| Tầng | Là gì | Ví dụ thành phần |
| --- | --- | --- |
| Language model | Hàm dự đoán token tiếp theo | Phân phối xác suất trên từ vựng |
| LLM | Language model quy mô lớn (thường dùng kiến trúc Transformer) | Trọng số đã huấn luyện (base hoặc instruct) |
| Chatbot | Sản phẩm dùng LLM | LLM + system prompt + lịch sử hội thoại + giao diện + công cụ (tìm kiếm, chạy mã) + bộ lọc an toàn |

### Minh họa: bốn tác vụ

| Tác vụ | Prompt ví dụ | Đầu ra mong đợi |
| --- | --- | --- |
| Hỏi đáp | “Thủ đô của Nhật Bản là gì?” | “Tokyo.” |
| Tóm tắt | “Tóm tắt đoạn sau trong 1 câu: \[đoạn văn\]” | Một câu tóm tắt |
| Dịch | “Dịch sang tiếng Anh: Hôm nay trời đẹp.” | “The weather is nice today.” |
| Sinh mã | “Viết hàm Python kiểm tra số nguyên tố.” | Hàm `is_prime(n)` |

Một LLM có thể xử lý đồng thời cả bốn tác vụ trên mà không cần mô hình chuyên biệt cho từng bài toán.

## 2. Nền tảng kiến trúc

### Dàn ý

- 2.1 Transformer và vai trò của attention
- 2.2 Ba họ kiến trúc, vì sao LLM chọn decoder-only
- 2.3 Self-attention và causal mask
- 2.4 Luồng dữ liệu từ token đầu vào đến xác suất token tiếp theo

### Nội dung

Transformer (Vaswani và cộng sự, 2017) là kiến trúc mạng neural dựa trên cơ chế **attention**: mỗi vị trí trong chuỗi có thể “nhìn” các vị trí khác và quyết định nên chú ý bao nhiêu vào từng vị trí. Attention được kết hợp với mạng feed-forward, kết nối residual và chuẩn hóa. Khác với RNN xử lý tuần tự, Transformer cho phép xử lý song song các vị trí trong chuỗi khi huấn luyện.

Có ba họ kiến trúc: encoder-only (như BERT, giỏi hiểu và phân loại), encoder-decoder (như T5, giỏi dịch và tóm tắt), và **decoder-only** (như họ GPT, Llama), là họ phổ biến nhất cho LLM sinh văn bản vì mục tiêu huấn luyện đơn giản (dự đoán token tiếp theo) và mở rộng quy mô thuận lợi.

**Self-attention.** Mỗi token được biến thành ba vector: Query (Q), Key (K), Value (V). Điểm chú ý giữa hai vị trí là tích vô hướng giữa Q và K, sau đó softmax thành trọng số và dùng trọng số đó lấy tổng có trọng số của V:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

*(trong đó $d_k$ là số chiều của vector key)*

**Causal attention** thêm mặt nạ M để token ở vị trí t chỉ nhìn được các vị trí từ 1 đến t, không nhìn được tương lai. Với chuỗi 4 token, ma trận cho phép (1 = được nhìn):

```
        t1 t2 t3 t4
t1      1  0  0  0
t2      1  1  0  0
t3      1  1  1  0
t4      1  1  1  1
```

Vì sao cần mặt nạ này? Khi huấn luyện, mô hình dự đoán token t+1 từ các token ≤ t tại *mọi* vị trí cùng lúc. Nếu được nhìn trước token đích thì mô hình sẽ “chép đáp án”. Mặt nạ vừa chặn việc đó, vừa cho phép huấn luyện song song trên cả chuỗi.

### Minh họa: sơ đồ luồng dữ liệu

```
Văn bản → Tokenizer → token ID → Embedding + thông tin vị trí (positional encoding hoặc RoPE, tùy kiến trúc)
   → [ Khối Transformer × N ]
        mỗi khối: Masked self-attention → cộng residual + chuẩn hóa
                  → Mạng feed-forward   → cộng residual + chuẩn hóa
   → Lớp tuyến tính cuối (kích thước = |V|) → logits
   → Softmax → xác suất của từng token trong từ vựng cho vị trí kế tiếp
```

Tại mỗi vị trí, mô hình xuất một phân phối xác suất trên toàn bộ từ vựng; trong quá trình sinh tự hồi quy, chỉ phân phối tại vị trí cuối cùng được sử dụng để lấy mẫu token tiếp theo.

## 3. Tokenization

### Dàn ý

- 3.1 Vì sao không dùng từ hay ký tự làm đơn vị
- 3.2 Subword và BPE
- 3.3 Token ID và bảng embedding
- 3.4 Tokenize thực tế: tiếng Việt và tiếng Anh

### Nội dung

Mô hình chỉ xử lý số, nên văn bản phải được chia thành **token** rồi ánh xạ sang **token ID**. Dùng nguyên từ thì từ vựng quá lớn và gặp từ lạ (out-of-vocabulary); dùng ký tự thì chuỗi quá dài. Giải pháp phổ biến là **subword**, điển hình là **BPE (Byte Pair Encoding)**: bắt đầu từ ký tự (hoặc byte), lặp lại việc gộp cặp đơn vị xuất hiện nhiều nhất trong dữ liệu thành một đơn vị mới, cho đến khi đủ cỡ từ vựng.

**Từ token ID đến embedding.** Mô hình có ma trận embedding $E \in \mathbb{R}^{|V| \times d}$ ($|V|$ là cỡ từ vựng, $d$ là số chiều). Token có ID = $i$ thì vector đầu vào là hàng thứ $i$ của $E$. Các vector này được huấn luyện cùng mô hình, nên các token có vai trò tương tự thường có xu hướng có vector gần nhau, nhưng không có gì bảo đảm điều này luôn đúng. Cũng cần phân biệt embedding đầu vào (một vector học được cho mỗi token ID) với biểu diễn theo ngữ cảnh, được tạo ra sau khi đi qua các lớp Transformer.

**Lưu ý về tiếng Việt.** Số token của tiếng Việt phụ thuộc vào tokenizer cụ thể và dữ liệu dùng để xây tokenizer đó. Với một số tokenizer tối ưu chủ yếu cho tiếng Anh, tiếng Việt có thể bị chia thành nhiều token hơn so với câu tiếng Anh có độ dài tương đương (nguyên nhân gồm tần suất trong dữ liệu, cách biểu diễn Unicode có dấu và thuật toán tách). Hệ quả có thể là tốn chi phí hơn và chiếm nhiều ngữ cảnh hơn.

### Minh họa: tokenize hai câu

```python
import tok_utils   # trong gói demo: src/tok_utils.py
E = tok_utils.load_gpt2()      # GPT-2 byte-level BPE (từ vựng 50.257)
SP = tok_utils.load_llama2()   # Llama 2 SentencePiece BPE (từ vựng 32.000)

sentences = [
    'Large language models learn to predict the next token.',
    'Mô hình ngôn ngữ lớn học cách dự đoán token tiếp theo.',
]
for s in sentences:
    ids, pieces = tok_utils.gpt2_tokenize(E, s)
    print(len(s), len(ids), ids)
    print(pieces)
    ids2, pieces2 = tok_utils.llama2_tokenize(SP, s)
    print(len(ids2), pieces2)
```

Ghi chú: một token tiếng Việt có thể chỉ là một phần byte của ký tự có dấu, nên khi decode từng token có thể hiện ký tự lỗi; đó là hành vi bình thường của byte-level BPE.

| Câu | Số ký tự | Số token | Token ID (rút gọn) | Các mảnh token |
| --- | --- | --- | --- | --- |
| Tiếng Anh | 54 | 10 | \[21968, 3303, 4981, 2193, 284, 4331, 262, 1306, 11241, 13\] | Large \| ␣language \| ␣models \| ␣learn \| ␣to \| ␣predict \| ␣the \| ␣next \| ␣token \| . |
| Tiếng Việt | 54 | 44 | \[44, 27083, 289, 127, 105, 77, 71, 23370, 27083, 77, …\] (44 ID) | M \| ô \| ␣h \| \[c3\] \| \[ac\] \| n \| h \| ␣ng \| ô \| n \| ␣ng \| \[e1\] \| \[bb\] \| \[af\] \| … (44 mảnh; \[xx\] là byte lẻ của một ký tự có dấu) |

Bảng trên dùng tokenizer GPT-2 (byte-level BPE, từ vựng 50.257). Với Llama 2 (SentencePiece, từ vựng 32.000), câu tiếng Anh ra 11 token và câu tiếng Việt ra 32 token. Nhận xét: ở GPT-2, câu tiếng Anh có 5,40 ký tự/token còn câu tiếng Việt chỉ 1,23 ký tự/token, tức cùng 54 ký tự nhưng tiếng Việt tốn gấp khoảng 4,4 lần số token. Trên 7 cặp câu song song (cùng nội dung ở hai ngôn ngữ), số token tiếng Việt chia số token tiếng Anh trung bình là 3,92 (GPT-2) và 2,65 (Llama 2). Nguyên nhân nhìn thấy trong các mảnh: nhiều ký tự có dấu (như ì, ữ, ớ) bị tách thành các byte lẻ, vì tokenizer GPT-2 vốn nhắm tới văn bản chủ yếu là tiếng Anh; Llama 2 ít bị tách hơn nhưng vẫn nhiều hơn tiếng Anh. Hệ quả là chi phí tính theo token tăng và ngữ cảnh bị chiếm nhanh hơn: một đoạn văn tiếng Việt mẫu chiếm 160 token so với 35 token cho đoạn tiếng Anh tương đương (GPT-2), nên cửa sổ 1.024 token chứa được khoảng 6,4 đoạn tiếng Việt so với 29,3 đoạn tiếng Anh. Đây là kết quả của hai tokenizer cụ thể trên 7 cặp câu, không khái quát cho mọi tokenizer.

## 4. Nguyên lý học

### Dàn ý

- 4.1 Next-token prediction là học tự giám sát
- 4.2 Dữ liệu huấn luyện
- 4.3 Hàm mất mát cross-entropy
- 4.4 Tạo cặp ngữ cảnh → token đích từ một câu

### Nội dung

**Next-token prediction.** Với chuỗi token $x_1, x_2, \dots, x_T$, mô hình được huấn luyện để dự đoán $x_t$ khi biết $x_1, \dots, x_{t-1}$. Đây là **học tự giám sát**: nhãn chính là token kế tiếp có sẵn trong văn bản, không cần con người gán nhãn.

**Dữ liệu huấn luyện.** Gồm văn bản từ web, sách, bài báo khoa học, mã nguồn... ở quy mô rất lớn. Dữ liệu thường được làm sạch, lọc chất lượng, loại trùng lặp và cân đối nguồn trước khi huấn luyện. Chất lượng và thành phần dữ liệu quyết định trực tiếp những gì mô hình biết và những thiên lệch nó mang theo (xem mục 9).

**Cross-entropy loss.** Tại mỗi vị trí, mô hình xuất phân phối xác suất $P$ trên từ vựng; mất mát là $-\log P(\text{token đúng})$. Mất mát trung bình trên chuỗi:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_1, \dots, x_{t-1})$$

Mô hình gán xác suất cao cho token đúng thì mất mát thấp, và ngược lại. **Perplexity** = $\exp(\mathcal{L}) = e^{\mathcal{L}}$, có thể hiểu là “số lựa chọn tương đương mà mô hình đang phân vân”.

Perplexity phụ thuộc vào tokenizer, dữ liệu và cách tính loss, do đó không so sánh trực tiếp giữa các mô hình sử dụng tokenizer khác nhau nếu điều kiện đánh giá không tương đương.

Ví dụ số: bốn vị trí, xác suất mô hình gán cho token đúng lần lượt là 0,2; 0,5; 0,9; 0,05.

```python
import numpy as np
p_target = np.array([0.2, 0.5, 0.9, 0.05])
loss = -np.log(p_target).mean()
print(loss, np.exp(loss))   # khoảng 1.351 và 3.86
```

Vị trí có xác suất 0,05 đóng góp mất mát lớn nhất (khoảng 3,0), cho thấy mô hình bị phạt nặng khi gán xác suất thấp cho token đúng. Mức xác suất 0,05 chỉ phản ánh mất mát lớn tại vị trí này chứ chưa đủ khẳng định mô hình dự đoán sai lệch hoàn toàn, do cần xem xét phân phối xác suất trên các token ứng viên còn lại.

### Minh họa: tạo cặp huấn luyện từ một câu

Câu: “Hà Nội là thủ đô của Việt Nam”. Để minh họa, giả sử mỗi từ là một token (trên thực tế là token của tokenizer):

| Ngữ cảnh (đầu vào) | Token mục tiêu |
| --- | --- |
| Hà | Nội |
| Hà Nội | là |
| Hà Nội là | thủ |
| Hà Nội là thủ | đô |
| Hà Nội là thủ đô | của |
| Hà Nội là thủ đô của | Việt |
| Hà Nội là thủ đô của Việt | Nam |

Một câu 8 token tạo ra 7 cặp huấn luyện. Nhờ causal mask, cả 7 cặp được tính trong *một* lần chạy xuôi.

Số liệu thật từ demo (MiniGPT mức ký tự, từ vựng 65 ký tự): đoán ngẫu nhiên có loss ln 65 ≈ 4,17; sau 2500 bước huấn luyện, loss trên tập kiểm định là 1,58 nats/ký tự (perplexity theo ký tự ≈ 4,85). Perplexity ở đây tính theo ký tự nên không so sánh được với perplexity theo token của mô hình khác.

## 5. Các giai đoạn huấn luyện

### Dàn ý

- 5.1 Pretraining
- 5.2 Supervised Fine-Tuning (SFT)
- 5.3 Preference alignment: ý tưởng RLHF và DPO
- 5.4 Sơ đồ và bảng tổng hợp

### Nội dung

**Pretraining (tiền huấn luyện).** Mục tiêu: next-token prediction trên kho văn bản khổng lồ. Kết quả là *base model*: có kiến thức ngôn ngữ và thế giới, nhưng chỉ giỏi “viết tiếp”, chưa chắc làm theo yêu cầu. Ví dụ hỏi “Thủ đô của Pháp là gì?”, base model có thể viết tiếp thêm một loạt câu hỏi khác thay vì trả lời.

**SFT (Supervised Fine-Tuning).** Huấn luyện tiếp trên tập nhỏ hơn nhiều gồm các cặp (chỉ dẫn, câu trả lời mẫu) chất lượng cao do con người viết hoặc tuyển chọn. Vẫn dùng cross-entropy, thường chỉ tính mất mát trên phần câu trả lời. Kết quả: mô hình biết cách đáp lại chỉ dẫn theo định dạng mong muốn.

**Preference alignment (căn chỉnh theo sở thích).** SFT chỉ dạy bằng ví dụ “đúng”. Nhiều khi cần dạy mô hình phân biệt câu trả lời *tốt hơn* và *kém hơn* (hữu ích, trung thực, an toàn). Dữ liệu ở đây là các bộ ba (prompt, câu trả lời được chọn, câu trả lời bị loại).

- **RLHF (ý tưởng):** (1) huấn luyện một *reward model* dự đoán con người thích câu trả lời nào; (2) dùng học tăng cường (thường là PPO) để tối ưu mô hình sao cho điểm reward cao, đồng thời có ràng buộc KL để không lệch quá xa mô hình SFT.
- **DPO (ý tưởng):** dùng trực tiếp các cặp dữ liệu ưa thích $(x, y_w, y_l)$ để tối ưu hóa xác suất của câu trả lời được chọn $y_w$ so với câu bị loại $y_l$ dựa trên mô hình tham chiếu $\pi_{\text{ref}}$, không cần reward model riêng và vòng lặp RL:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

### Minh họa: sơ đồ và bảng

```
Kho văn bản thô ──► [Pretraining] ──► Base model
                                         │
Cặp (chỉ dẫn, trả lời mẫu) ──► [SFT] ──► Instruct model
                                         │
Cặp (chosen, rejected) ──► [RLHF hoặc DPO] ──► Mô hình đã căn chỉnh
```

Quy trình ba giai đoạn trên là cấu trúc phổ biến của các mô hình chỉ dẫn hiện đại, tuy nhiên có thể được tinh chỉnh tùy theo mục tiêu kiến trúc cụ thể.

| Giai đoạn | Mục tiêu | Dữ liệu | Quy mô dữ liệu | Kết quả |
| --- | --- | --- | --- | --- |
| Pretraining | Dự đoán token tiếp theo | Văn bản thô | Rất lớn | Kiến thức và năng lực ngôn ngữ nền |
| SFT | Làm theo chỉ dẫn | (chỉ dẫn, trả lời mẫu) | Nhỏ hơn nhiều | Biết đối thoại, đúng định dạng |
| Alignment | Ưa thích câu trả lời tốt hơn | (prompt, chosen, rejected) | Nhỏ đến vừa | Hữu ích, an toàn, hợp ý người dùng hơn |

## 6. Quá trình sinh văn bản

### Dàn ý

- 6.1 Sinh tự hồi quy (autoregressive)
- 6.2 Greedy decoding
- 6.3 Temperature
- 6.4 Top-p (nucleus) sampling
- 6.5 Thử nghiệm cùng prompt, nhiều cấu hình

### Nội dung

**Autoregressive generation.** Mô hình sinh từng token một: (1) đưa prompt vào, lấy phân phối cho token kế tiếp; (2) chọn một token theo quy tắc giải mã; (3) nối token đó vào chuỗi; (4) lặp lại đến khi gặp token kết thúc hoặc đạt độ dài tối đa. Mỗi token sinh ra trở thành ngữ cảnh cho token sau, nên một lựa chọn sớm có thể ảnh hưởng toàn bộ phần còn lại.

**Greedy decoding:** luôn chọn token có xác suất cao nhất. Về nguyên tắc kết quả xác định (cùng mô hình, đầu vào và cấu hình thì chạy lại giống nhau; thực tế vẫn có thể lệch nhẹ do tính toán số học trên phần cứng), nhưng dễ lặp và thiếu đa dạng.

**Temperature (T):** chia logits $z_i$ cho $T$ trước hàm softmax:

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

$T < 1$ làm phân phối “nhọn” hơn (tập trung vào token xác suất cao, nhất quán hơn); $T > 1$ làm phân phối “phẳng” hơn (đa dạng hơn, có thể kém nhất quán). $T \to 0$ tương đương greedy decoding. Temperature không trực tiếp bảo đảm nội dung chính xác hay an toàn.

**Top-p (nucleus sampling):** chỉ giữ tập token nhỏ nhất $V^{(p)}$ có tổng xác suất tích lũy đạt tối thiểu ngưỡng $p$, chuẩn hóa lại rồi mới lấy mẫu:

$$\sum_{i \in V^{(p)}} P(x_i) \ge p$$

```python
import numpy as np
rng = np.random.default_rng(0)

def probs(logits, T=1.0):
    z = np.array(logits) / T
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()

def top_p_filter(p, top_p):
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = order[: np.searchsorted(cum, top_p) + 1]
    q = np.zeros_like(p)
    q[keep] = p[keep]
    return q / q.sum()

def sample(logits, T=1.0, top_p=1.0):
    return rng.choice(len(logits), p=top_p_filter(probs(logits, T), top_p))

logits = [2.0, 1.0, 0.5, -1.0]
for T in (0.3, 1.0, 2.0):
    print(T, probs(logits, T).round(3))
```

Với mô hình thật (Hugging Face), tham số tương ứng: `model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=60)`; greedy là `do_sample=False`.

### Minh họa: cùng một prompt, nhiều cấu hình

Prompt: “ROMEO:” rồi xuống dòng, chạy trên MiniGPT (GPT mức ký tự \~0,8 triệu tham số, tự huấn luyện trên văn bản Shakespeare; xem notebook demo). Mỗi lần sinh 160 ký tự; mỗi ô dưới đây là 100 ký tự đầu của mẫu với seed 0, 1, 2 (⏎ là xuống dòng). Cột nhận xét dùng số đo trên 30 mẫu mỗi cấu hình (greedy chỉ 3 mẫu vì xác định).

| Cấu hình | Lần 1 | Lần 2 | Lần 3 | Nhận xét (độ lặp lại, đa dạng, mạch lạc) |
| --- | --- | --- | --- | --- |
| Greedy | I will be so make the stand of the sent of the sent ⏎ That the sent of the sent of the senter of the s… | (giống hệt lần 1) | (giống hệt lần 1) | 3 lần giống hệt nhau. Lặp vòng rõ (cụm “of the sent” lặp lại): 40,5% bộ ba từ bị lặp; 5,1% “từ” không có trong dữ liệu. |
| T = 0,3; top-p = 1 | There have here not so shall be so did the strange ⏎ That you have here been the seems of my serves, ⏎ A… | What shall say the grace of my lord. ⏎⏎ CAPULET: ⏎ The virtue of the state of his face of the state, ⏎ And… | O, my lord; and the prince of the present of the best court. ⏎⏎ CORIOLANUS: ⏎ What we would make the suc… | 30/30 mẫu khác nhau nhưng khá đơn điệu, nhiều cụm “of the …”. Ít từ lạ nhất (1,7%); distinct-2 = 0,885; chỉ 1,8% bộ ba từ bị lặp. |
| T = 0,8; top-p = 0,95 | Thou must there not made a means by the georn, ⏎ And prayer his true receitor of such life. ⏎⏎ PRINCE ED… | What speak so me us leave my eye, ⏎ And shame with a virtue countiness in this poor ⏎ all the present of… | Of my lord; I'll return them more to such with all mischard. ⏎ Thou shalt not be this boy! and let's t… | Đa dạng cao (distinct-2 = 0,994), đọc ra giống kịch Shakespeare nhưng có từ bịa (“georn”, “receitor”). Từ lạ 5,9%; cùng T = 0,8 mà không dùng top-p là 9,7%. |
| T = 1,5; top-p = 1 | thy farms kiss selter? ⏎ How nevery youd Ixels fagMeyour wigh framentre his ⏎ death being such one conte… | But, ⏎ 'tis repembuut leave atment, a king. ⏎⏎ BUCKINGHBLA: ⏎ Sacric herm, infil. O, a half, and friens'd,… | Od my lord; each, you'll! ⏎ TonguouNt by my noble wretchcought, ho! ⏎⏎ MENENIUS: ⏎ Rats Robpeam: ⏎ Go prison… | Mất mạch lạc: 38,2% “từ” không có trong dữ liệu; nhiều từ ghép vô nghĩa và tên người nói bịa (“BUCKINGHBLA”); 47% mẫu có tên người nói không có trong dữ liệu. |

Số liệu đo đầy đủ (cùng prompt, 160 ký tự mỗi mẫu, seed 0–29):

| Cấu hình | Số mẫu | Tỉ lệ “từ” không có trong dữ liệu | Distinct-2 (theo từ) | Bộ ba từ bị lặp |
| --- | --- | --- | --- | --- |
| Greedy | 3 | 5,1% | 0,474 | 40,5% |
| T = 0,3 | 30 | 1,7% | 0,885 | 1,8% |
| T = 0,8 | 30 | 9,7% | 0,995 | 0,0% |
| T = 0,8; top-p = 0,95 | 30 | 5,9% | 0,994 | 0,0% |
| T = 1,0 | 30 | 16,0% | 0,998 | 0,1% |
| T = 1,5 | 30 | 38,2% | 1,000 | 0,0% |

Kết quả: Chiến lược greedy decoding tạo ra các mẫu xác định giống hệt nhau nhưng dễ rơi vào vòng lặp ngữ nghĩa. Nhiệt độ $T$ càng tăng thì độ đa dạng từ vựng càng lớn nhưng tỷ lệ xuất hiện từ không có trong ngữ liệu cũng gia tăng tương ứng, làm giảm tính mạch lạc cú pháp. Cấu hình kết hợp $T = 0,8$ và top-$p = 0,95$ giúp giảm tỷ lệ từ lạ từ 9,7% xuống 5,9% trong khi vẫn duy trì độ phong phú từ vựng. Mô hình MiniGPT mức ký tự được sử dụng nhằm mục đích minh họa xu hướng định tính của các cơ chế giải mã; các giá trị định lượng cụ thể không phản ánh năng lực của các LLM quy mô lớn.

## 7. Khả năng của LLM

### Dàn ý

- 7.1 Làm theo chỉ dẫn
- 7.2 In-context learning
- 7.3 Các tác vụ ngôn ngữ
- 7.4 So sánh zero-shot và few-shot

### Nội dung

**Làm theo chỉ dẫn (instruction following)** chủ yếu đến từ SFT và alignment (mục 5): mô hình hiểu yêu cầu viết bằng ngôn ngữ tự nhiên như “tóm tắt”, “dịch”, “liệt kê”.

**In-context learning (ICL):** mô hình học cách làm một tác vụ từ chính các ví dụ đặt trong prompt, *không cập nhật trọng số*. Việc “học” là sự thích nghi dựa trên nội dung ngữ cảnh được đưa vào; nó không đồng nghĩa với việc mô hình học thêm kiến thức vào tham số.

- **Zero-shot:** chỉ có mô tả tác vụ, không có ví dụ.
- **Few-shot:** mô tả tác vụ kèm vài ví dụ mẫu (đầu vào → đầu ra).

LLM xử lý được nhiều tác vụ ngôn ngữ: phân loại, trích xuất thông tin, tóm tắt, dịch, hỏi đáp, viết lại văn bản, sinh mã.

### Minh họa: phân loại cảm xúc review tiếng Việt

**Zero-shot:**

```
Phân loại cảm xúc của câu sau thành đúng một trong ba nhãn: tích cực, tiêu cực, trung tính.
Chỉ trả lời bằng đúng một nhãn, không giải thích.

Câu: "Giao hàng chậm nhưng sản phẩm dùng rất ổn."
Nhãn:
```

**Few-shot (6 ví dụ):**

```
Phân loại cảm xúc của câu sau thành đúng một trong ba nhãn: tích cực, tiêu cực, trung tính.
Chỉ trả lời bằng đúng một nhãn, không giải thích.

Câu: "Quán ăn này sạch sẽ, món nào cũng ngon, phục vụ chu đáo."
Nhãn: tích cực

Câu: "Đặt hàng hai tuần mà vẫn chưa nhận được, liên hệ cũng không ai trả lời."
Nhãn: tiêu cực

Câu: "Chiếc áo có ba cỡ là S, M và L."
Nhãn: trung tính

Câu: "Chất vải mỏng quá, giặt một lần đã nhăn nhúm."
Nhãn: tiêu cực

Câu: "Cuốn sách gồm 12 chương và dày khoảng 300 trang."
Nhãn: trung tính

Câu: "Màn hình sáng rõ, loa to, dùng thích hơn tôi nghĩ."
Nhãn: tích cực

Câu: "Giao hàng chậm nhưng sản phẩm dùng rất ổn."
Nhãn:
```

Thiết lập thử nghiệm có kiểm soát: Đánh giá trên cùng bộ 20 câu kiểm tra tiếng Việt, mô hình Qwen 2.5 7B, nhiệt độ cố định T = 0. Biến can thiệp duy nhất là việc bổ sung 6 ví dụ mẫu trong prompt (few-shot). Các ví dụ mẫu hoàn toàn độc lập với tập kiểm tra; nhãn đối sánh (ground truth) được xác định trước theo quy tắc gán nhãn thống nhất (ưu tiên mệnh đề chính sau liên từ đối lập). Kết quả ghi nhận độ chính xác và mức độ tuân thủ định dạng đầu ra:

| Cách prompt | Số câu đúng / 20 | Sai định dạng | Nhận xét |
| --- | --- | --- | --- |
| Zero-shot | 16 / 20 (80,0%) | 0 / 20 (0%) | Đạt độ chính xác tuyệt đối trên tập câu dễ (13/13) và 42,9% trên tập câu khó (3/7). Sai 4 câu khó (mỉa mai, phủ định kép, hỗn hợp khen-chê); cả 4 trường hợp đều bị dự đoán lệch sang "trung tính". Không xảy ra lỗi định dạng. |
| Few-shot (6 ví dụ) | 17 / 20 (85,0%) | 0 / 20 (0%) | Cải thiện độ chính xác tổng thể (+5%) nhờ phân loại đúng câu mỉa mai ("Pin tuyệt vời thật..." → tiêu cực), nâng độ chính xác tập khó lên 57,1% (4/7). Tiếp tục nhầm 3 câu hỗn hợp và phủ định kép sang "trung tính". Không xảy ra lỗi định dạng. |

Về mặt lý thuyết và thực nghiệm, few-shot hỗ trợ định hình phong cách phản hồi và giải quyết các ca biên ngữ nghĩa, tuy nhiên độ cải thiện phụ thuộc vào chất lượng, tính đại diện và độ cân bằng nhãn của các ví dụ mẫu.

## 8. Cách thích nghi và bổ sung tri thức

### Dàn ý

- 8.1 Prompting
- 8.2 Fine-tuning (kể cả LoRA)
- 8.3 RAG
- 8.4 Bảng so sánh và cách chọn

### Nội dung

**Prompting:** thay đổi cách đặt yêu cầu (chỉ dẫn rõ hơn, ví dụ few-shot, định dạng đầu ra) mà không đổi mô hình. Nhanh và rẻ nhất, nhưng bị giới hạn bởi độ dài ngữ cảnh và không thêm được tri thức lớn.

**Fine-tuning:** huấn luyện thêm để thay đổi trọng số, dùng khi cần phong cách, định dạng hoặc hành vi ổn định trên một tác vụ. Có thể cập nhật toàn bộ tham số, hoặc dùng các phương pháp tiết kiệm như **LoRA** (chỉ huấn luyện các ma trận hạng thấp bổ sung). Cần dữ liệu có nhãn chất lượng; tri thức mới đưa vào bằng fine-tuning khó kiểm soát và khó cập nhật.

**RAG (Retrieval-Augmented Generation):** khi có câu hỏi, hệ thống tìm các đoạn tài liệu liên quan trong kho tri thức rồi đưa chúng vào prompt để mô hình trả lời dựa trên đó.

```
Tài liệu → tiền xử lý → chia đoạn → chỉ mục tìm kiếm (vector, từ khóa hoặc kết hợp)
Câu hỏi → truy xuất các đoạn liên quan (tùy chọn: xếp hạng lại)
        → ghép vào prompt cùng câu hỏi → LLM sinh câu trả lời (có thể kèm trích dẫn)
```

### Minh họa: bảng so sánh

| Tiêu chí | Prompting | Fine-tuning | RAG |
| --- | --- | --- | --- |
| Thay đổi trọng số | Không | Có | Không |
| Dữ liệu cần có | Vài ví dụ hoặc chỉ dẫn | Tập (đầu vào, đầu ra) chất lượng, quy mô phù hợp với tác vụ (không có số mẫu tối thiểu chung) | Kho tài liệu được chia đoạn và lập chỉ mục |
| Chi phí ban đầu | Rất thấp | Cao nhất (dữ liệu, tính toán, đánh giá) | Trung bình (xây chỉ mục, hệ thống truy xuất) |
| Cập nhật tri thức | Phải sửa prompt | Phải huấn luyện lại | Chỉ cần cập nhật kho tài liệu |
| Truy vết nguồn | Khó | Khó | Dễ (trích đoạn nguồn) |
| Phù hợp khi | Thử nhanh, tác vụ chung | Cần phong cách/định dạng ổn định, tác vụ chuyên biệt lặp lại nhiều | Hỏi đáp trên tài liệu nội bộ, tri thức hay thay đổi |
| Hạn chế | Giới hạn ngữ cảnh, kém ổn định | Tốn kém, có thể làm giảm năng lực khác | Chất lượng phụ thuộc truy xuất; vẫn có thể sinh sai |

Cách chọn thực dụng: bắt đầu bằng prompting; nếu thiếu tri thức riêng hoặc tri thức hay đổi thì thêm RAG; nếu cần hành vi hoặc giọng văn ổn định mà prompt không đạt được thì cân nhắc fine-tuning. Ba cách có thể kết hợp.

## 9. Hạn chế và đánh giá

### Dàn ý

- 9.1 Hallucination
- 9.2 Thiên lệch
- 9.3 Giới hạn ngữ cảnh và mốc tri thức
- 9.4 Chất lượng phụ thuộc tác vụ và dữ liệu
- 9.5 Ví dụ câu trả lời sai và cách kiểm chứng

### Nội dung

**Hallucination (ảo giác):** mô hình sinh ra nội dung nghe rất hợp lý nhưng sai hoặc bịa (số liệu, trích dẫn, tên người, tài liệu không tồn tại). Nguyên nhân gốc: mục tiêu dự đoán token tiếp theo không trực tiếp tối ưu tính đúng sự thật của mọi phát biểu, nên mô hình có thể sinh nội dung trôi chảy nhưng sai (retrieval, công cụ và kiểm chứng giúp giảm rủi ro nhưng không loại bỏ hoàn toàn); khi thiếu thông tin nó vẫn có xu hướng “điền cho trọn câu”.

**Thiên lệch (bias):** mô hình học từ dữ liệu do con người tạo ra nên có thể phản ánh và khuếch đại định kiến về giới, vùng miền, văn hóa; đồng thời hiệu năng thường kém hơn ở các ngôn ngữ và lĩnh vực ít dữ liệu.

**Giới hạn ngữ cảnh:** mô hình chỉ xử lý được một số lượng token nhất định trong một lần. Nội dung vượt quá giới hạn bị cắt hoặc không dùng được; ngay cả trong giới hạn, nghiên cứu cho thấy mô hình có thể dùng kém thông tin nằm ở giữa ngữ cảnh dài. Ngoài ra mô hình có *mốc tri thức*: kiến thức phụ thuộc vào dữ liệu và thời điểm huấn luyện, nên mô hình không mặc nhiên có thông tin về các sự kiện xảy ra sau đó; muốn cập nhật cần công cụ tìm kiếm hoặc RAG.

**Chất lượng phụ thuộc tác vụ và dữ liệu:** một mô hình có thể hoạt động tốt ở tác vụ diễn đạt hoặc viết lại văn bản nhưng kém ở tính toán nhiều bước hoặc câu hỏi chuyên sâu hiếm gặp. Do đó, mô hình cần được đánh giá trên đúng tác vụ và miền dữ liệu mục tiêu, thay vì suy diễn từ ấn tượng tổng quát.

### Minh họa: kiểm chứng hiện tượng ảo giác (hallucination)

Hiện tượng ảo giác thường xuất hiện ở các tác vụ truy vấn thông tin hẹp, tính toán đa bước hoặc thao tác ở cấp độ ký tự (do cơ chế subword tokenization). Bảng dưới đây ghi nhận trường hợp thực nghiệm trên mô hình Qwen 2.5 7B (temperature = 0):

| Mục | Nội dung |
| --- | --- |
| Prompt | “Trong từ 'strawberry' có bao nhiêu chữ r?” |
| Câu trả lời của mô hình | “Trong tiếng Việt, từ "strawberry" có 2 chấm (.) khi được viết hoa và dấu chấm sau cùng. Tuy nhiên, nếu bạn đang hỏi về số lượng chấm trong từ này khi được viết bằng tiếng Việt, thì câu trả lời là không có chấm cả, vì "strawberry" được viết nguyên âm trong tiếng Việt không cần thêm dấu chấm.” |
| Chỗ sai | Mô hình không đếm số ký tự "r" mà tự suy diễn lạc đề sang “dấu chấm (.)”, đồng thời tự bịa ra các quy tắc ngữ pháp tiếng Việt phi lý cho từ tiếng Anh. Thực tế từ "strawberry" có 3 chữ 'r' (st**r**awbe**rr**y). |
| Cách kiểm chứng đã dùng | Tách chuỗi ký tự thủ công và kiểm tra đối chiếu bằng mã Python: `sum(1 for c in 'strawberry'.lower() if c == 'r')` |
| Kết quả kiểm chứng | Từ "strawberry" có chính xác 3 chữ 'r'. Mô hình gặp hiện tượng ảo giác (hallucination) do cơ chế tokenizer tách từ thành các mảnh subword thay vì nhìn thấy từng ký tự đơn lẻ, kết hợp với việc suy diễn tự động của mô hình tự hồi quy khi gặp câu hỏi đếm ký tự. |

Quy trình kiểm chứng thông tin:

1. Tách câu trả lời thành từng khẳng định kiểm chứng được.
2. Đối chiếu với nhiều nguồn độc lập, đáng tin cậy (cơ sở dữ liệu học thuật, trang chính thức, sách giáo khoa).
3. Nếu mô hình đưa trích dẫn hoặc đường dẫn, mở ra kiểm tra xem có thật và có đúng nội dung không.
4. Không xem việc mô hình trả lời nhất quán qua nhiều lần hỏi là bằng chứng đúng: mô hình có thể lặp lại cùng một thông tin sai.
5. Với tính toán và mã, chạy lại bằng công cụ thật.
6. Dùng RAG hoặc công cụ tìm kiếm để cung cấp nguồn, và với quyết định quan trọng phải có con người xem xét.

## 10. Demo và kết luận

### Dàn ý

- 10.1 Thiết kế thử nghiệm có kiểm soát
- 10.2 Bảng kết quả
- 10.3 Ví dụ đầu ra
- 10.4 Phân tích lỗi
- 10.5 Nhận xét và kết luận

### Nội dung

**Thiết kế thử nghiệm có kiểm soát:** Thực nghiệm đánh giá năng lực suy luận và mức độ nhạy cảm prompt bằng cách thay đổi duy nhất yếu tố ví dụ ngữ cảnh (zero-shot so với few-shot), giữ cố định mô hình, siêu tham số và dữ liệu đầu vào:

| Thành phần | Thiết lập thực nghiệm |
| --- | --- |
| Câu hỏi nghiên cứu | Ví dụ few-shot có cải thiện phân loại cảm xúc tiếng Việt so với zero-shot không? |
| Biến can thiệp | Phương pháp prompt: zero-shot vs few-shot (6 ví dụ mẫu) |
| Yếu tố kiểm soát | Mô hình Qwen 2.5 7B, temperature = 0, cùng bộ 20 câu kiểm tra, cùng cơ chế chấm điểm |
| Chỉ số đo lường | Tỷ lệ chính xác tổng thể, tỷ lệ chính xác theo độ khó, số phản hồi sai định dạng |
| Dữ liệu | 20 câu review tiếng Việt phân loại theo độ khó (dễ: 13 câu, khó: 7 câu gồm mỉa mai, phủ định kép, hỗn hợp) |

Tập dữ liệu 20 câu kiểm tra đóng vai trò tập mẫu có kiểm soát để quan sát định tính và định lượng hành vi của mô hình; các kết luận cần được diễn giải trong phạm vi quy mô này.

### Bảng kết quả

Kết quả thử nghiệm phân loại cảm xúc zero-shot vs few-shot trên mô hình thật Qwen 2.5 7B (bộ 20 câu kiểm tra, nhiệt độ T = 0):

| Cấu hình | Đúng / 20 | Tỷ lệ đúng | Đúng tập dễ (13 câu) | Đúng tập khó (7 câu) | Sai định dạng |
| --- | --- | --- | --- | --- | --- |
| Zero-shot | 16 / 20 | 80,0% | 13 / 13 (100%) | 3 / 7 (42,9%) | 0 / 20 (0%) |
| Few-shot (6 ví dụ) | 17 / 20 | 85,0% | 13 / 13 (100%) | 4 / 7 (57,1%) | 0 / 20 (0%) |

**Ma trận nhầm lẫn (hàng = nhãn đúng, cột = dự đoán):**

- **Few-shot:**
  ```
  pred        tiêu cực  trung tính  tích cực
  gold
  tiêu cực           6           0         0
  trung tính         0           6         0
  tích cực           0           3         5
  ```
  *(Tiêu cực đúng 6/6 = 100%, Trung tính đúng 6/6 = 100%, Tích cực đúng 5/8 = 62,5% do 3 câu bị nhầm sang trung tính)*

- **Zero-shot:**
  ```
  pred        tiêu cực  trung tính  tích cực
  gold
  tiêu cực           5           1         0
  trung tính         0           6         0
  tích cực           0           3         5
  ```
  *(Tiêu cực đúng 5/6 = 83,3% do câu mỉa mai bị nhầm sang trung tính, Trung tính đúng 6/6 = 100%, Tích cực đúng 5/8 = 62,5% do 3 câu nhầm sang trung tính)*

### Ví dụ đầu ra

Bốn câu tiêu biểu đại diện cho câu dễ, câu khó được sửa đúng nhờ few-shot, và các câu khó cả hai chế độ đều chưa làm được:

| Câu kiểm tra | Nhãn đúng | Zero-shot | Few-shot | Ghi chú |
| --- | --- | --- | --- | --- |
| "Điện thoại dùng mượt, pin trâu, rất đáng tiền." (câu 1) | tích cực | tích cực (Đúng) | tích cực (Đúng) | Dễ: khen trực diện, từ ngữ tích cực rõ ràng |
| "Pin tuyệt vời thật, sạc đầy dùng được đúng hai tiếng." (câu 11) | tiêu cực | trung tính (Sai) | tiêu cực (Đúng) | Khó (mỉa mai): few-shot giúp mô hình nhận diện giọng điệu mỉa mai |
| "Giao hàng chậm nhưng sản phẩm dùng rất ổn." (câu 5) | tích cực | trung tính (Sai) | trung tính (Sai) | Khó (hỗn hợp): có cả chê lẫn khen, mô hình thiên về chọn an toàn |
| "Không thể nói là không hài lòng với dịch vụ này." (câu 19) | tích cực | trung tính (Sai) | trung tính (Sai) | Khó (phủ định kép): cấu trúc phức tạp khiến mô hình đánh giá nước đôi |

### Phân tích lỗi

**1. Phân tích lỗi thử nghiệm LLM phân loại cảm xúc (dựa trên `results/llm_errors.csv`):**

Tổng cộng có 7 lượt dự đoán sai (4 lượt ở zero-shot và 3 lượt ở few-shot), chia thành các nhóm nguyên nhân:

- **Ý kiến lẫn lộn (câu 5 và câu 20):**
  - *Ví dụ:* “Giao hàng chậm nhưng sản phẩm dùng rất ổn” (câu 5) và “Hỏng hóc linh tinh nhưng bên bảo hành xử lý rất nhanh nên tôi vẫn yên tâm” (câu 20).
  - *Nhãn đúng:* tích cực (theo quy tắc gán nhãn: ưu tiên vế cảm xúc chính sau từ “nhưng”).
  - *Dự đoán của mô hình:* trung tính ở cả zero-shot và few-shot.
  - *Giả thuyết nguyên nhân:* Mô hình gặp xung đột khi văn bản chứa đồng thời tín hiệu tiêu cực (“chậm”, “hỏng hóc linh tinh”) và tích cực (“rất ổn”, “rất nhanh”, “yên tâm”). Do không có cơ chế suy luận thứ bậc ngữ nghĩa sâu hoặc do 6 ví dụ few-shot chưa chứa ví dụ phân giải liên từ "nhưng", mô hình chọn giải pháp trung hòa an toàn là "trung tính".
- **Mỉa mai, ẩn dụ (câu 11):**
  - *Ví dụ:* “Pin tuyệt vời thật, sạc đầy dùng được đúng hai tiếng.”
  - *Nhãn đúng:* tiêu cực.
  - *Dự đoán của mô hình:* zero-shot đoán "trung tính" (sai); few-shot đoán "tiêu cực" (đúng).
  - *Giả thuyết nguyên nhân:* Zero-shot bị đánh lừa bởi từ cảm thán tích cực bề mặt (“tuyệt vời thật”) kết hợp với thông số khách quan (“hai tiếng”). Khi có 6 ví dụ few-shot định hướng, mô hình kích hoạt tốt hơn năng lực liên kết tri thức thực tế (thời lượng pin 2 tiếng là quá ngắn) để nhận ra ngữ nghĩa mỉa mai ngầm.
- **Phủ định kép (câu 19):**
  - *Ví dụ:* “Không thể nói là không hài lòng với dịch vụ này.”
  - *Nhãn đúng:* tích cực.
  - *Dự đoán của mô hình:* trung tính ở cả zero-shot và few-shot.
  - *Giả thuyết nguyên nhân:* Cấu trúc phủ định kép (“không thể nói là không...”) làm tăng độ phức tạp cú pháp. Đặc biệt, tokenizer tiếng Việt tách các từ phủ định thành nhiều token riêng lẻ, khiến mô hình khó nắm bắt mối liên kết logic giữa hai tầng phủ định để suy ra cảm xúc tích cực, dẫn đến phán đoán nước đôi là "trung tính".
- **Lỗi định dạng:** Cả zero-shot và few-shot đều đạt **0 lỗi** (20/20 câu đều trả về đúng duy nhất 1 nhãn), cho thấy khả năng tuân thủ định dạng chỉ dẫn (instruction following) của Qwen 2.5 7B rất hoàn hảo.

**2. Phân tích lỗi thử nghiệm giải mã (MiniGPT, mục 6):**

| Cấu hình | Có “từ” không tồn tại | Lặp vòng | Tên người nói không có trong dữ liệu |
| --- | --- | --- | --- |
| Greedy (3 mẫu) | 100% | 100% | 0% |
| T = 0,3 | 47% | 0% | 7% |
| T = 0,8 | 100% | 0% | 23% |
| T = 0,8; top-p = 0,95 | 97% | 0% | 13% |
| T = 1,0 | 100% | 0% | 23% |
| T = 1,5 | 100% | 0% | 47% |

Nhận xét: lỗi chuyển từ “lặp vòng” (greedy) sang “bịa chi tiết” (T cao). Ví dụ lỗi bịa là tên người nói “BUCKINGHBLA” ở T = 1,5, không có trong dữ liệu huấn luyện. Đây là giả thuyết đã kiểm tra bằng cách tra từ và tên trong dữ liệu gốc, nhưng chỉ áp dụng cho mô hình nhỏ mức ký tự này.

### Kết luận

1. LLM là mô hình ngôn ngữ quy mô lớn; phần lớn LLM sinh văn bản hiện đại dùng kiến trúc Transformer, trong đó decoder-only là phổ biến, thường được huấn luyện bằng dự đoán token tiếp theo với cross-entropy.
2. Ba giai đoạn huấn luyện (pretraining, SFT, alignment) lần lượt tạo kiến thức nền, khả năng làm theo chỉ dẫn và hành vi phù hợp mong muốn.
3. Đầu ra phụ thuộc vào cách giải mã (greedy, temperature, top-p) và cách viết prompt (zero-shot, few-shot).
4. Prompting, fine-tuning và RAG giải quyết các nhu cầu khác nhau, thường kết hợp với nhau.
5. Mô hình có thể sai một cách tự tin, mang thiên lệch và bị giới hạn bởi ngữ cảnh; vì vậy luôn cần kiểm chứng và đánh giá trên đúng tác vụ.
6. Kết quả thử nghiệm thực tế: Thử nghiệm phân loại cảm xúc 20 câu tiếng Việt trên mô hình Qwen 2.5 7B (temperature = 0) cho thấy few-shot (6 ví dụ) nâng độ chính xác từ 80,0% (16/20) lên 85,0% (17/20), giúp sửa đúng câu mỉa mai ở tập khó (độ chính xác tập khó tăng từ 42,9% lên 57,1%); cả hai chế độ đều đạt 100% câu dễ (13/13) và 0 lỗi định dạng. Điểm hạn chế chính của mô hình nằm ở các ca biên: câu phủ định kép và câu hỗn hợp khen-chê thường bị phân loại an toàn về "trung tính". Giới hạn thực nghiệm: cỡ mẫu nhỏ (20 câu), thực hiện trên một mô hình (Qwen 2.5 7B) và một tác vụ cụ thể.

## Tài liệu tham khảo

- Vaswani et al. (2017). Attention Is All You Need.
- Sennrich et al. (2016). Neural Machine Translation of Rare Words with Subword Units (BPE).
- Brown et al. (2020). Language Models are Few-Shot Learners (GPT-3).
- Ouyang et al. (2022). Training language models to follow instructions with human feedback (InstructGPT, RLHF).
- Rafailov et al. (2023). Direct Preference Optimization.
- Holtzman et al. (2020). The Curious Case of Neural Text Degeneration (nucleus sampling).
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Liu et al. (2023). Lost in the Middle: How Language Models Use Long Contexts.
