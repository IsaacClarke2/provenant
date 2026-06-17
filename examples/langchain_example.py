"""Provenant + LangChain — provenance-rerank a vector store's results.

Tag each document's metadata with an `origin` at ingest time, then rerank the
`(Document, score)` pairs from `similarity_search_with_score`. Runs standalone on
faked Documents.

    pip install git+https://github.com/IsaacClarke2/provenant
    python examples/langchain_example.py
"""
from provenant import MODEL_GENERATED, USER
from provenant.adapters import rerank_lc_documents


class _Doc:  # stand-in for langchain_core.documents.Document
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


# --- AT INGEST: tag origin in metadata ---------------------------------------
#   Document(page_content="...", metadata={"origin": USER})
#
# --- AT QUERY: one line ------------------------------------------------------
#   pairs = vectorstore.similarity_search_with_score(query, k=20)
#   docs  = rerank_lc_documents(pairs)        # provenance-reranked Documents


def _demo():
    # (Document, distance) pairs — langchain distances are smaller-is-better
    pairs = [
        (_Doc("Your meeting is on Tuesday", {"origin": MODEL_GENERATED}), 0.18),
        (_Doc("Your meeting is on Thursday", {"origin": USER}), 0.24),
        (_Doc("You booked room 4B", {"origin": USER}), 0.30),
    ]
    print("vector order (by distance, smaller=closer):")
    for doc, dist in sorted(pairs, key=lambda p: p[1]):
        print(f"  d={dist:.2f} [{doc.metadata['origin']:<15}] {doc.page_content}")

    print("\nprovenance-reranked:")
    for doc in rerank_lc_documents(pairs):
        print(f"         [{doc.metadata['origin']:<15}] {doc.page_content}")


if __name__ == "__main__":
    _demo()
