"""Sentence embeddings with ONNX Runtime: no PyTorch, no sentence-transformers (decision L3).

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), 384 dims,
using the quantized ONNX export the model repo ships (onnx/model_quint8_avx2.onnx, about a
quarter of the 470 MB full model). Pooling is done here: mean over tokens weighted by the
attention mask, then L2-normalised, which is what the model's 1_Pooling config specifies.

The Docker build downloads the model once into EMBED_MODEL_DIR (see `download`); nothing is
downloaded at run time. Without the model the service falls back to keyword retrieval.

    python -m app.embed download [dir]     # fetch model files (Docker build, or locally)
    python -m app.embed selftest [dir]     # sanity check: related phrases closer than unrelated
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = os.getenv("EMBED_REVISION", "main")
ONNX_FILE = os.getenv("EMBED_ONNX_FILE", "onnx/model_quint8_avx2.onnx")
FILES = [ONNX_FILE, "tokenizer.json"]
DEFAULT_DIR = Path(os.getenv("EMBED_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "minilm"))
MAX_TOKENS = 128


def model_available(model_dir: Path = DEFAULT_DIR) -> bool:
    return all((model_dir / Path(f).name).exists() for f in FILES)


def download(model_dir: Path = DEFAULT_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = model_dir / Path(f).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"have {dest}")
            continue
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{f}"
        print(f"downloading {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


class Embedder:
    dim = 384

    def __init__(self, model_dir: Path = DEFAULT_DIR, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = f"{REPO.split('/')[-1]}:{Path(ONNX_FILE).name}"
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        pad = next((t for t in ("<pad>", "[PAD]") if self.tok.token_to_id(t) is not None), None)
        self.tok.enable_padding(pad_id=self.tok.token_to_id(pad) if pad else 0, pad_token=pad or "[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / Path(ONNX_FILE).name), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        outs = [o.name for o in self.sess.get_outputs()]
        self.output = "sentence_embedding" if "sentence_embedding" in outs else outs[0]

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i : i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            res = self.sess.run([self.output], {k: v for k, v in feed.items() if k in self.inputs})[0]
            if res.ndim == 3:  # token embeddings -> masked mean pooling
                m = mask[..., None].astype(np.float32)
                res = (res * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(res.astype(np.float32))
        vecs = np.vstack(out)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        return vecs


def selftest(model_dir: Path = DEFAULT_DIR) -> None:
    """Fail loudly if the model doesn't behave like a paraphrase model (wrong file, bad pooling)."""
    e = Embedder(model_dir)
    v = e.encode(["대박이야", "대박이다", "학교에 갔다", "绝绝子", "真的绝绝子", "数学是一门学科"])
    ko_close, ko_far = float(v[0] @ v[1]), float(v[0] @ v[2])
    zh_close, zh_far = float(v[3] @ v[4]), float(v[3] @ v[5])
    print(f"{e.name}: dim {v.shape[1]} · ko related {ko_close:.2f} vs unrelated {ko_far:.2f} · "
          f"zh related {zh_close:.2f} vs unrelated {zh_far:.2f}")
    assert v.shape[1] == Embedder.dim, "unexpected embedding size"
    assert ko_close > ko_far and zh_close > zh_far, "related phrases are not closer than unrelated ones"
    print("selftest OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DIR
    if cmd == "download":
        download(d)
    elif cmd == "selftest":
        selftest(d)
    else:
        raise SystemExit(__doc__)
