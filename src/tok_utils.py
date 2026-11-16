"""Tokenizer thật dùng trong demo mục 3: GPT-2 BPE (byte-level BPE) và Llama 2 (SentencePiece BPE).
File từ vựng nằm trong thư mục data/ (nguồn: bản công khai trên GitHub của OpenAI/GPT-2 và karpathy/llama2.c)."""
import importlib.util
import json
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_gpt2():
    spec = importlib.util.spec_from_file_location("gpt2_bpe_openai", ROOT / "src" / "gpt2_bpe_openai.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    enc = json.load(open(DATA / "gpt2_encoder.json", encoding="utf-8"))
    merges = [tuple(l.split()) for l in open(DATA / "gpt2_vocab.bpe", encoding="utf-8").read().split("\n")[1:-1]]
    return m.Encoder(enc, merges)


def gpt2_tokenize(E, text):
    """Trả về (ids, pieces). Mảnh không phải UTF-8 hợp lệ (byte lẻ của ký tự có dấu) hiển thị dạng <xx xx>."""
    ids = E.encode(text)
    pieces = []
    for i in ids:
        b = bytes(E.byte_decoder[c] for c in E.decoder[i])
        try:
            pieces.append(b.decode("utf-8"))
        except UnicodeDecodeError:
            pieces.append("<" + " ".join(f"{x:02x}" for x in b) + ">")
    assert E.decode(ids) == text  # mã hóa -> giải mã phải khôi phục đúng văn bản
    return ids, pieces


def load_llama2():
    return spm.SentencePieceProcessor(model_file=str(DATA / "llama2_tokenizer.model"))


def llama2_tokenize(sp, text):
    ids = sp.encode(text)
    return ids, sp.id_to_piece(ids)
