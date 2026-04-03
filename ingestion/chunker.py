"""
chunker.py — Hierarchical document chunking using LlamaIndex.

Uses HierarchicalNodeParser to create parent + child nodes that preserve
section header context. Child nodes (small chunks) are what gets embedded;
parent nodes are stored for context window retrieval.

Input:  list of Document dicts (from scraper) or documents.json path
Output: chunks.json — list of chunk dicts ready for embedding
"""

import json
from pathlib import Path
from typing import Union

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Chunk sizes: parent (large context) → child (what gets embedded)
# These sizes work well for formal government/regulatory text
CHUNK_SIZES = [2048, 512, 128]


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Chunk a list of document dicts.

    Returns:
        leaf_chunks  — small chunks for embedding (list of dicts)
        all_chunks   — all nodes including parents (list of dicts), for context retrieval
    """
    # Convert raw dicts to LlamaIndex Document objects
    llama_docs = []
    for doc in documents:
        llama_docs.append(
            LlamaDocument(
                text=doc["text"],
                metadata={
                    "source_url": doc["source_url"],
                    "title": doc["title"],
                    "doc_type": doc["doc_type"],
                },
                # Prevent metadata fields from being embedded (they'd pollute vectors)
                excluded_embed_metadata_keys=["source_url", "doc_type"],
                excluded_llm_metadata_keys=["source_url", "doc_type"],
            )
        )

    # Build hierarchical nodes (large → medium → small)
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=CHUNK_SIZES)
    all_nodes: list[BaseNode] = parser.get_nodes_from_documents(llama_docs)

    # Leaf nodes are the smallest chunks — these get embedded
    leaf_nodes = get_leaf_nodes(all_nodes)

    print(f"Total nodes (all levels): {len(all_nodes)}")
    print(f"Leaf nodes (for embedding): {len(leaf_nodes)}")

    def node_to_dict(node: BaseNode, is_leaf: bool) -> dict:
        return {
            "node_id": node.node_id,
            "text": node.get_content(),
            "metadata": node.metadata,
            "parent_node_id": (
                node.parent_node.node_id if node.parent_node else None
            ),
            "is_leaf": is_leaf,
            # Keep the source document title in every chunk for citation
            "title": node.metadata.get("title", ""),
            "source_url": node.metadata.get("source_url", ""),
        }

    leaf_node_ids = {n.node_id for n in leaf_nodes}
    all_chunks = [node_to_dict(n, n.node_id in leaf_node_ids) for n in all_nodes]
    leaf_chunks = [c for c in all_chunks if c["is_leaf"]]

    return leaf_chunks, all_chunks


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_chunks(leaf_chunks: list[dict], all_chunks: list[dict]) -> None:
    leaf_path = PROCESSED_DATA_DIR / "leaf_chunks.json"
    all_path = PROCESSED_DATA_DIR / "all_chunks.json"

    with open(leaf_path, "w", encoding="utf-8") as f:
        json.dump(leaf_chunks, f, ensure_ascii=False, indent=2)

    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(leaf_chunks)} leaf chunks → {leaf_path}")
    print(f"Saved {len(all_chunks)} total chunks  → {all_path}")


def load_raw_documents(path: Union[str, Path, None] = None) -> list[dict]:
    if path is None:
        path = RAW_DATA_DIR / "documents.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_chunker(documents: list[dict] | None = None) -> list[dict]:
    """
    Run chunker. If documents is None, loads from data/raw/documents.json.
    Returns leaf_chunks list.
    """
    if documents is None:
        print("Loading raw documents from disk...")
        documents = load_raw_documents()

    print(f"Chunking {len(documents)} documents...")
    leaf_chunks, all_chunks = chunk_documents(documents)
    save_chunks(leaf_chunks, all_chunks)
    return leaf_chunks


if __name__ == "__main__":
    run_chunker()
