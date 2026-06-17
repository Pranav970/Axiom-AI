"""Semantic-chunking ingestion CLI.

Usage:
    python -m app.vectorstore.ingest --path ./data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore.store import get_embeddings, get_vectorstore


SUPPORTED_EXTS = {".txt", ".md", ".markdown"}


def _load_documents(root: Path) -> List[Document]:
    if not root.exists():
        raise FileNotFoundError(f"Data path does not exist: {root}")

    files: List[Path] = []
    if root.is_file():
        files = [root]
    else:
        for ext in SUPPORTED_EXTS:
            files.extend(root.rglob(f"*{ext}"))

    if not files:
        raise FileNotFoundError(
            f"No supported documents ({SUPPORTED_EXTS}) found under {root}"
        )

    docs: List[Document] = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = fp.read_text(encoding="latin-1")
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(fp.resolve()), "filename": fp.name},
                )
            )
    return docs


def _semantic_split(docs: List[Document]) -> List[Document]:
    """Semantic chunking with a recursive fallback safety net.

    SemanticChunker can occasionally emit very large chunks for documents
    without strong semantic boundaries — we cap with a recursive splitter
    to guarantee bounded chunk sizes.
    """
    embeddings = get_embeddings()
    semantic = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90.0,
    )
    semantic_chunks = semantic.split_documents(docs)

    safety = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_chunks = safety.split_documents(semantic_chunks)

    # Attach stable chunk ids for citation
    for idx, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('filename','doc')}::{idx}"
    return final_chunks


def ingest(path: str, reset: bool = False) -> int:
    settings = get_settings()
    root = Path(path).resolve()

    docs = _load_documents(root)
    chunks = _semantic_split(docs)

    vs = get_vectorstore()
    if reset:
        try:
            vs.delete_collection()
        except Exception:
            pass
        # Re-instantiate after deletion
        from app.vectorstore.store import get_vectorstore as _gv
        _gv.cache_clear()  # type: ignore[attr-defined]
        vs = _gv()

    vs.add_documents(chunks)
    print(
        f"[ingest] {len(docs)} document(s) -> {len(chunks)} chunks "
        f"persisted to {settings.chroma_persist_dir} "
        f"(collection='{settings.chroma_collection}')"
    )
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--path", default="./data", help="File or directory to ingest.")
    parser.add_argument("--reset", action="store_true", help="Drop existing collection first.")
    args = parser.parse_args()
    try:
        ingest(args.path, reset=args.reset)
        return 0
    except Exception as e:
        print(f"[ingest][error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
