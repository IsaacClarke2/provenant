# Contributing

Thanks for looking. Provenant is small on purpose.

## Dev setup

```bash
git clone https://github.com/IsaacClarke2/provenant
cd provenant
pip install -e ".[dev]"
python -m pytest -q          # 27 tests, < 1s
```

## Design laws (please keep them)

- **No hardcoded behavior.** Origin is classified structurally (by write-site),
  never by parsing the meaning of text. No regex, no keyword lists.
- **Continuous & asymptotic.** Weights move smoothly and stay strictly inside an
  open interval — never an absolute 0 or 1.
- **Traceable.** No `except: return default` that hides errors; every weight and
  ranking decision exposes its terms (`explain`, `score_with_provenance`).
- **Zero runtime dependencies in the core.** Heavy things (a real embedder for
  the benchmark) go behind the `[bench]` extra.

## Good first contributions

- Adapters for more stores (Letta, Chroma, Qdrant, LlamaIndex).
- A real LongMemEval/LOCOMO run on better hardware (see
  `provenant/bench/longmemeval.py`) — the full 500-question number.
- Origin classifiers for specific agent frameworks' write-sites.

Open an issue first for anything bigger than an adapter.
