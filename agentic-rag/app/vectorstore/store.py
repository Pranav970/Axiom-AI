"""ChromaDB factory — single source of truth for the vector store."""
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def get_retriever(k: int | None = None) -> VectorStoreRetriever:
    settings = get_settings()
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k or settings.retrieval_k},
    )
