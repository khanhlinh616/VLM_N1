"""Huấn luyện MiniGPT trên tinyshakespeare (pretraining theo next-token prediction).

Chạy:  python train_minigpt.py [--steps 2500]        (~16-18 phút trên 1 CPU)
Có thể ngắt và chạy lại: tiến trình được lưu ở checkpoints/train_state.pt và tự tiếp tục.
Dùng --max-seconds N để chỉ chạy N giây mỗi lần (mã thoát 3 nếu chưa xong).
Kết quả: checkpoints/minigpt.pt, results/train_history.json
"""
import argparse, json, sys, torch
sys.path.insert(0, "src")
from minigpt import CharTokenizer, MiniGPT, train

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2500)
ap.add_argument("--max-seconds", type=float, default=None)
args = ap.parse_args()

torch.set_num_threads(1)
text = open("data/tinyshakespeare.txt", encoding="utf-8").read()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
torch.manual_seed(0)
model = MiniGPT(tok.vocab_size)
print(f"tham số: {model.n_params():,} | vocab: {tok.vocab_size} | train/val ký tự: {len(train_data):,}/{len(val_data):,}", flush=True)
hist, done = train(model, train_data, val_data, steps=args.steps, log=lambda s: print(s, flush=True),
                   state_path="checkpoints/train_state.pt", max_seconds=args.max_seconds)
if not done:
    print("chưa xong, chạy lại để tiếp tục", flush=True); sys.exit(3)
torch.save({"state": model.state_dict(), "cfg": model.cfg, "chars": tok.chars}, "checkpoints/minigpt.pt")
json.dump({"steps": args.steps, "n_params": model.n_params(), "history": hist}, open("results/train_history.json", "w"), indent=1)
print("xong", flush=True)
