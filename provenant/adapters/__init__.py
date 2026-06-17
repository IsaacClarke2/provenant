"""Thin, dependency-free adapters for popular memory stores.

Provenant ships as a *neutral adapter* — it works WITH your existing store
(mem0, LangChain, Letta, a raw vector DB), never as a competing store. These
helpers are duck-typed: they import nothing from those libraries, so Provenant
stays a zero-heavy-dependency package and the adapters work against whatever
result shape your version returns.
"""
from .mem0 import rerank_mem0_results
from .langchain import rerank_lc_documents

__all__ = ["rerank_mem0_results", "rerank_lc_documents"]
