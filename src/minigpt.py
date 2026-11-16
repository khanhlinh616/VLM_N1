"""MiniGPT: GPT nhỏ (decoder-only, causal attention) mức ký tự, dùng để minh họa.

Kiến trúc giống các LLM decoder-only (embedding + vị trí -> N khối
[masked self-attention + MLP] -> logits -> softmax) nhưng rất nhỏ để huấn luyện
được trên CPU. Đây KHÔNG phải một LLM thực sự: chỉ để quan sát cơ chế.
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.hs = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        # mặt nạ nhân quả: vị trí t chỉ nhìn được các vị trí <= t
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x, return_att=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.drop(self.proj(y))
        return (y, att) if return_att else y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x, return_att=False):
        if return_att:
            a, att = self.attn(self.ln1(x), return_att=True)
            x = x + a
            return x + self.mlp(self.ln2(x)), att
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd,
                        n_head=n_head, n_layer=n_layer, dropout=dropout)
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, return_att=False):
        B, T = idx.shape
        assert T <= self.block_size, f"chuỗi dài {T} vượt quá cửa sổ ngữ cảnh {self.block_size}"
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        atts = []
        for blk in self.blocks:
            if return_att:
                x, a = blk(x, return_att=True)
                atts.append(a)
            else:
                x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_att:
            return logits, loss, atts
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- giải mã
def adjust_probs(logits, temperature=1.0, top_p=1.0):
    """logits (V,) -> xác suất sau khi áp dụng temperature và top-p."""
    z = logits / max(temperature, 1e-8)
    p = F.softmax(z, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(p, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = (cum - sp) < top_p  # giữ token cho đến khi tổng tích lũy >= top_p
        sp = sp * keep
        p = torch.zeros_like(p).scatter(0, si, sp)
        p = p / p.sum()
    return p


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_p=1.0, greedy=False, generator=None):
    """Sinh tự hồi quy. idx: (1,T). Trả về (1, T+max_new_tokens)."""
    model.eval()
    for _ in range(max_new_tokens):
        ctx = idx[:, -model.block_size:]
        logits, _ = model(ctx)
        last = logits[0, -1]
        if greedy:
            nxt = last.argmax().view(1, 1)
        else:
            p = adjust_probs(last, temperature, top_p)
            nxt = torch.multinomial(p, 1, generator=generator).view(1, 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# ------------------------------------------------------------ huấn luyện
def get_batch(data, block_size, batch_size, gen):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, gen, iters=20):
    model.eval()
    ls = [model(*get_batch(data, block_size, batch_size, gen))[1].item() for _ in range(iters)]
    model.train()
    return sum(ls) / len(ls)


def train(model, train_data, val_data, steps, batch_size=32, lr=3e-3, warmup=100, eval_every=250,
          seed=0, log=print, state_path=None, max_seconds=None):
    """Huấn luyện có thể tiếp tục (resume) từ state_path. Trả về (history, finished)."""
    import os
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    start, hist, base_sec = 0, [], 0.0
    if state_path and os.path.exists(state_path):
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start, hist, base_sec = st["step"], st["hist"], st["seconds"]
        log(f"[resume] tiếp tục từ bước {start}")
    t0 = time.time()
    model.train()

    def save(step):
        if state_path:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist,
                        "seconds": base_sec + time.time() - t0}, state_path)

    for step in range(start, steps + 1):
        if step < warmup:
            cur = lr * (step + 1) / warmup
        else:
            cur = 0.1 * lr + 0.9 * lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = cur
        if (step % eval_every == 0 or step == steps) and not (hist and hist[-1]["step"] == step):
            ge = torch.Generator().manual_seed(10_000_000 + step)
            tr = estimate_loss(model, train_data, model.block_size, batch_size, ge)
            va = estimate_loss(model, val_data, model.block_size, batch_size, ge)
            sec = base_sec + time.time() - t0
            hist.append(dict(step=step, train_loss=tr, val_loss=va, seconds=round(sec, 1)))
            log(f"step {step:5d} | train {tr:.4f} | val {va:.4f} | {sec:6.1f}s")
            save(step)
        if step == steps:
            break
        if max_seconds and time.time() - t0 > max_seconds:
            save(step)
            return hist, False
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)   # mỗi bước một seed -> resume tái lập được
        x, y = get_batch(train_data, model.block_size, batch_size, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return hist, True
