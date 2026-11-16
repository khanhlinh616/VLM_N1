"""Thí nghiệm có kiểm soát: zero-shot vs few-shot trên cùng tác vụ phân loại cảm xúc tiếng Việt.

Biến thay đổi DUY NHẤT: có / không có ví dụ trong prompt.
Giữ cố định: mô hình, temperature=0, bộ 20 câu test, cách chấm điểm.

Cách chạy (chọn MỘT backend):

  1) Claude API (cần biến môi trường ANTHROPIC_API_KEY):
       python run_llm_experiment.py --backend anthropic --model <tên-model>

  2) Server tương thích OpenAI (Ollama, LM Studio, vLLM...), ví dụ Ollama:
       python run_llm_experiment.py --backend openai_compat --model qwen2.5:7b \
              --base-url http://localhost:11434/v1

  3) Không có API: dán tay vào giao diện chat rồi chấm điểm:
       python run_llm_experiment.py --backend manual --export          # tạo results/manual_prompts.csv
       (điền cột 'output' bằng câu trả lời của mô hình, mỗi prompt trong MỘT cuộc chat mới)
       python run_llm_experiment.py --backend manual --score results/manual_prompts.csv --model "<tên model>"

  4) Kiểm tra pipeline (KHÔNG phải kết quả LLM, chỉ là luật từ khóa):
       python run_llm_experiment.py --backend mock
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from datetime import date

import pandas as pd

LABELS = ["tích cực", "tiêu cực", "trung tính"]
INSTRUCTION = (
    "Phân loại cảm xúc của câu sau thành đúng một trong ba nhãn: tích cực, tiêu cực, trung tính.\n"
    "Chỉ trả lời bằng đúng một nhãn, không giải thích."
)


def load_data():
    test = pd.read_csv("data/test_sentiment_vi.csv")
    shots = pd.read_csv("data/fewshot_examples.csv")
    overlap = set(test.sentence) & set(shots.sentence)
    assert not overlap, f"Ví dụ few-shot trùng bộ test: {overlap}"
    return test, shots


def build_prompt(sentence, mode, shots):
    if mode == "zero_shot":
        return f'{INSTRUCTION}\n\nCâu: "{sentence}"\nNhãn:'
    ex = "\n\n".join(f'Câu: "{r.sentence}"\nNhãn: {r.label}' for r in shots.itertuples())
    return f'{INSTRUCTION}\n\n{ex}\n\nCâu: "{sentence}"\nNhãn:'


def parse_label(raw):
    """Trả về (nhãn, đúng_định_dạng). Đúng định dạng = đầu ra chỉ gồm đúng một nhãn."""
    t = (raw or "").strip().lower().strip(" .\n\t\"'*`")
    if t in LABELS:
        return t, True
    found = [l for l in LABELS if l in t]
    if len(found) == 1:
        return found[0], False  # có nhãn nhưng kèm lời thừa
    return "không hợp lệ", False


# ------------------------------------------------------------------ backends
def _post(url, headers, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call_anthropic(prompt, model, **_):
    key = os.environ["ANTHROPIC_API_KEY"]
    out = _post("https://api.anthropic.com/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01"},
                {"model": model, "max_tokens": 20, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
    return "".join(b.get("text", "") for b in out["content"])


def call_openai_compat(prompt, model, base_url="http://localhost:11434/v1", **_):
    key = os.environ.get("OPENAI_API_KEY", "none")
    out = _post(base_url.rstrip("/") + "/chat/completions", {"Authorization": f"Bearer {key}"},
                {"model": model, "max_tokens": 20, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
    return out["choices"][0]["message"]["content"]


def call_mock(prompt, **_):
    """Luật từ khóa rất thô, CHỈ để kiểm tra pipeline. Không phải LLM."""
    last = prompt.rsplit("Câu:", 1)[-1].lower()
    neg = ["hỏng", "chậm", "không giống", "treo", "nguội", "thất vọng", "không tương xứng"]
    pos = ["tốt", "đẹp", "ngon", "thân thiện", "hay", "ổn", "sạch", "đáng tiền", "dễ hiểu", "hợp lý", "mượt"]
    if any(w in last for w in neg):
        return "tiêu cực"
    if any(w in last for w in pos):
        return "tích cực"
    return "trung tính"


BACKENDS = {"anthropic": call_anthropic, "openai_compat": call_openai_compat, "mock": call_mock}


# ------------------------------------------------------------------- chấm điểm
def score(pred_df, prefix):
    pred_df["pred"], pred_df["valid_format"] = zip(*pred_df["output"].map(parse_label))
    pred_df["correct"] = pred_df["pred"] == pred_df["gold"]
    pred_df.to_csv(f"results/{prefix}predictions.csv", index=False)
    rows = []
    for mode, g in pred_df.groupby("mode"):
        rows.append(dict(mode=mode, n=len(g), accuracy=round(g.correct.mean(), 3), n_correct=int(g.correct.sum()),
                         acc_easy=round(g[g.difficulty == "dễ"].correct.mean(), 3),
                         acc_hard=round(g[g.difficulty == "khó"].correct.mean(), 3),
                         format_ok=int(g.valid_format.sum()), format_bad=int((~g.valid_format).sum())))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"results/{prefix}summary.csv", index=False)
    err = pred_df[~pred_df.correct][["mode", "id", "sentence", "gold", "pred", "difficulty", "note", "output"]]
    err.to_csv(f"results/{prefix}errors.csv", index=False)
    print("\n=== KẾT QUẢ ===")
    print(summ.to_string(index=False))
    print("\n=== Ma trận nhầm lẫn (hàng = nhãn đúng, cột = dự đoán) ===")
    for mode, g in pred_df.groupby("mode"):
        print(f"[{mode}]")
        print(pd.crosstab(g.gold, g.pred).to_string())
    print(f"\nCác câu sai: {len(err)}  -> results/{prefix}errors.csv")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(BACKENDS) + ["manual"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--export", action="store_true", help="(manual) xuất prompt ra CSV")
    ap.add_argument("--score", default="", help="(manual) CSV đã điền cột output")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    test, shots = load_data()
    prefix = "_mock_" if args.backend == "mock" else "llm_"
    if args.backend == "mock":
        print("!!! MOCK: luật từ khóa, KHÔNG phải kết quả của LLM. Chỉ dùng để thử pipeline. !!!")

    if args.backend == "manual":
        if args.export:
            rows = [dict(id=r.id, mode=m, prompt=build_prompt(r.sentence, m, shots), output="")
                    for m in ("zero_shot", "few_shot") for r in test.itertuples()]
            pd.DataFrame(rows).to_csv("results/manual_prompts.csv", index=False, quoting=csv.QUOTE_ALL)
            print("Đã tạo results/manual_prompts.csv (40 prompt). Điền cột 'output' rồi chạy lại với --score.")
            return
        man = pd.read_csv(args.score).fillna("")
        df = man.merge(test, on="id")[["id", "mode", "sentence", "gold", "difficulty", "note", "output"]]
    else:
        fn = BACKENDS[args.backend]
        rows = []
        for mode in ("zero_shot", "few_shot"):
            for r in test.itertuples():
                prompt = build_prompt(r.sentence, mode, shots)
                out = fn(prompt, model=args.model, base_url=args.base_url)
                rows.append(dict(id=r.id, mode=mode, sentence=r.sentence, gold=r.gold, difficulty=r.difficulty,
                                 note=r.note if isinstance(r.note, str) else "", output=out))
                time.sleep(0.05)
        df = pd.DataFrame(rows)

    score(df, prefix)
    json.dump(dict(backend=args.backend, model=args.model, date=str(date.today()), temperature=0,
                   n_test=len(test), n_shots=len(shots)), open(f"results/{prefix}meta.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
