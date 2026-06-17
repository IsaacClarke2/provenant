"""LongMemEval retrieval benchmark — the real, dataset-backed launch number.

This is the concrete implementation behind the `LongMemEvalAdapter` stub: it
loads the official LongMemEval haystack, retrieves with a real embedder, tags
each turn's origin by its conversational role (user turns are grounded; assistant
turns are model-generated), and measures what provenance reranking does to
retrieval — with no LLM and no faked numbers.

Why this is a fair test of Provenant: in LongMemEval the answer is something the
*user* stated, so gold-evidence turns are overwhelmingly user-origin (~94% in
the oracle split). Assistant turns are fluent, on-topic distractors that a plain
cosine retriever routinely pulls into the top-k. Provenant demotes them.

Metrics (mean over questions), reranking OFF vs ON:
  * gold_recall@k     - a gold (has_answer) turn is in the top-k
  * gold_mrr          - reciprocal rank of the first gold turn
  * assistant_in_topk - fraction of the top-k that are assistant-origin turns

Requires the `bench` extra: `pip install fastembed numpy` and the dataset file
from HuggingFace `xiaowu0162/longmemeval` (`longmemeval_s` or `longmemeval_oracle`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..origins import MODEL_GENERATED, USER
from ..rerank import provenance_rerank


@dataclass(frozen=True)
class Turn:
    text: str
    origin: str       # USER or MODEL_GENERATED, from the conversational role
    is_gold: bool      # has_answer in the dataset
    cosine: float = 0.0


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    turns: List[Turn]


def _iter_json_array(path: str, limit: Optional[int] = None):
    """Stream top-level objects from a big JSON array, stopping after `limit`.

    The LongMemEval file is ~278 MB; `json.load` would balloon to multiple GB.
    This reads in chunks and decodes one object at a time (no extra dependency),
    so memory stays bounded — critical on a small, shared box.
    """
    dec = json.JSONDecoder()
    buf = ""
    started = False
    count = 0
    with open(path, "r") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB
            if chunk:
                buf += chunk
            if not started:
                buf = buf.lstrip()
                if buf[:1] == "[":
                    buf = buf[1:]
                    started = True
                elif not chunk:
                    return
                else:
                    continue
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                if buf[0] == ",":
                    buf = buf[1:]
                    continue
                if buf[0] == "]":
                    return
                try:
                    obj, idx = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    break  # incomplete object — need more data
                buf = buf[idx:]
                count += 1
                yield obj
                if limit is not None and count >= limit:
                    return
            if not chunk:
                return


def load_longmemeval(path: str, limit: Optional[int] = None) -> List[Question]:
    """Load LongMemEval entries, flattening each haystack into role-tagged turns."""
    out: List[Question] = []
    for e in _iter_json_array(path, limit):
        turns: List[Turn] = []
        for session in e.get("haystack_sessions", []):
            for t in session:
                role = t.get("role")
                content = (t.get("content") or "").strip()
                if not content:
                    continue
                turns.append(
                    Turn(
                        text=content,
                        origin=USER if role == "user" else MODEL_GENERATED,
                        is_gold=bool(t.get("has_answer")),
                    )
                )
        if turns:
            out.append(Question(qid=e["question_id"], text=e["question"], turns=turns))
    return out


# --- default real embedder: fastembed (public package, not Mate) -----------------
def fastembed_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    threads: Optional[int] = None,
) -> Callable[[Sequence[str]], "object"]:
    """Return embed(texts)->np.ndarray (L2-normalized) backed by fastembed.

    `threads` caps onnxruntime's thread pool — set it to 1-2 on a shared/low-core
    box so the embedder does not starve co-located services (e.g. a live app).
    """
    import numpy as np
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name, threads=threads)

    def embed(texts: Sequence[str]):
        # parallel=0 disables fastembed's internal multiprocessing — otherwise it
        # forks a worker per CPU, each reloading the model (memory x N) and
        # spiking load on a small box. Single process + a modest batch keeps the
        # peak bounded, which matters when co-located with a live service.
        vecs = np.array(
            list(model.embed(list(texts), parallel=0, batch_size=64)),
            dtype=np.float32,
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    return embed


def _rank_indices(turns: Sequence[Turn], use_provenance: bool) -> List[int]:
    idx = list(range(len(turns)))
    if not use_provenance:
        return sorted(idx, key=lambda i: -turns[i].cosine)
    ranked = provenance_rerank(
        idx,
        score_of=lambda i: turns[i].cosine,
        origin_of=lambda i: turns[i].origin,
    )
    return ranked


def _eval_question(turns: Sequence[Turn], k: int, use_provenance: bool) -> Dict[str, float]:
    order = _rank_indices(turns, use_provenance)
    top = order[:k]
    recall = 1.0 if any(turns[i].is_gold for i in top) else 0.0
    # reciprocal rank of the first gold turn over the full ranking
    mrr = 0.0
    for rank, i in enumerate(order, 1):
        if turns[i].is_gold:
            mrr = 1.0 / rank
            break
    asst = sum(1 for i in top if turns[i].origin == MODEL_GENERATED) / max(1, len(top))
    return {"recall": recall, "mrr": mrr, "assistant_in_topk": asst}


@dataclass(frozen=True)
class LMEReport:
    n_questions: int
    k: int
    model: str
    recall_base: float
    recall_prov: float
    mrr_base: float
    mrr_prov: float
    assistant_base: float
    assistant_prov: float

    def as_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def run_longmemeval(
    path: str,
    limit: Optional[int] = None,
    k: int = 5,
    embed: Optional[Callable[[Sequence[str]], "object"]] = None,
    model_name: str = "BAAI/bge-small-en-v1.5",
    threads: Optional[int] = None,
    progress: bool = False,
) -> LMEReport:
    """Run the LongMemEval retrieval benchmark, reranking OFF vs ON."""
    import numpy as np

    questions = load_longmemeval(path, limit)
    embed = embed or fastembed_embedder(model_name, threads=threads)

    agg = {m: {"base": 0.0, "prov": 0.0} for m in ("recall", "mrr", "assistant_in_topk")}
    for qi, q in enumerate(questions):
        qvec = embed([q.text])[0]
        tvecs = embed([t.text for t in q.turns])
        cosines = tvecs @ qvec  # both L2-normalized -> cosine similarity
        scored = [
            Turn(text=t.text, origin=t.origin, is_gold=t.is_gold, cosine=float(c))
            for t, c in zip(q.turns, cosines)
        ]
        base = _eval_question(scored, k, use_provenance=False)
        prov = _eval_question(scored, k, use_provenance=True)
        for m in agg:
            agg[m]["base"] += base[m]
            agg[m]["prov"] += prov[m]
        if progress:
            print(f"  [{qi + 1}/{len(questions)}] {q.qid}", flush=True)

    n = max(1, len(questions))
    return LMEReport(
        n_questions=len(questions),
        k=k,
        model=model_name,
        recall_base=agg["recall"]["base"] / n,
        recall_prov=agg["recall"]["prov"] / n,
        mrr_base=agg["mrr"]["base"] / n,
        mrr_prov=agg["mrr"]["prov"] / n,
        assistant_base=agg["assistant_in_topk"]["base"] / n,
        assistant_prov=agg["assistant_in_topk"]["prov"] / n,
    )


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Provenant x LongMemEval retrieval benchmark")
    ap.add_argument("--data", required=True, help="path to longmemeval_s or longmemeval_oracle")
    ap.add_argument("--limit", type=int, default=None, help="number of questions (default: all)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=None,
                    help="cap embedder threads (use 1-2 on a shared/low-core box)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    rep = run_longmemeval(args.data, limit=args.limit, k=args.k,
                          threads=args.threads, progress=args.progress)
    d = rep.as_dict()
    print("\nProvenant x LongMemEval (retrieval, real embedder, no LLM)")
    print(f"  questions: {d['n_questions']}   k: {d['k']}   embedder: {d['model']}")
    print(f"  gold recall@{d['k']}       base {d['recall_base']:.3f}  ->  provenance {d['recall_prov']:.3f}"
          f"   (Δ {d['recall_prov'] - d['recall_base']:+.3f})")
    print(f"  gold MRR              base {d['mrr_base']:.3f}  ->  provenance {d['mrr_prov']:.3f}"
          f"   (Δ {d['mrr_prov'] - d['mrr_base']:+.3f})")
    print(f"  assistant in top-{d['k']}    base {d['assistant_base']:.3f}  ->  provenance {d['assistant_prov']:.3f}"
          f"   (Δ {d['assistant_prov'] - d['assistant_base']:+.3f})")
    if args.out:
        json.dump(d, open(args.out, "w"), indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    _main()
