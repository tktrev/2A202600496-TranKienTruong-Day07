from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker, compute_similarity
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    # "data/python_intro.txt",
    # "data/vector_store_notes.md",
    # "data/rag_system_design.md",
    # "data/customer_support_playbook.txt",
    # "data/chunking_experiment_report.md",
    # "data/vi_retrieval_notes.md",
    "data/vinbus.md",
]


CHUNKER_TYPES = ("recursive", "fixed", "sentence")


def load_documents_from_files(
    file_paths: list[str],
    chunk_size: int = 500,
    chunker_type: str = "recursive",
) -> list[Document]:
    """Load and chunk documents from file paths.

    Args:
        file_paths: List of file paths to load.
        chunk_size: Maximum chunk size in characters (used by 'recursive' and 'fixed').
        chunker_type: Chunking strategy – one of 'recursive', 'fixed', or 'sentence'.
    """
    if chunker_type not in CHUNKER_TYPES:
        raise ValueError(f"chunker_type must be one of {CHUNKER_TYPES}, got {chunker_type!r}")

    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    if chunker_type == "fixed":
        chunker = FixedSizeChunker(chunk_size=chunk_size)
    elif chunker_type == "sentence":
        chunker = SentenceChunker()
    else:
        chunker = RecursiveChunker(chunk_size=chunk_size)

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        chunks = chunker.chunk(content)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{path.stem}_chunk_{i}",
                    content=chunk,
                    metadata={"source": str(path), "extension": path.suffix.lower(), "chunk_index": i},
                )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """LLM using OpenAI for manual RAG testing."""
    load_dotenv(override=False)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def run_manual_demo(
    question: str | None = None,
    sample_files: list[str] | None = None,
    chunker_type: str = "recursive",
) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print(f"Chunker: {chunker_type}")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files, chunker_type=chunker_type)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0


def score_answer(candidate: str, reference: str, embedder=None) -> float:
    """Cosine similarity between embeddings of candidate and reference answers.

    Falls back to token-overlap F1 if embedder is unavailable.
    """
    if embedder is not None and callable(embedder):
        try:
            vec_a = embedder(candidate)
            vec_b = embedder(reference)
            return compute_similarity(vec_a, vec_b)
        except Exception:
            pass
    # Fallback: token-overlap F1
    candidate_tokens = set(candidate.lower().split())
    reference_tokens = set(reference.lower().split())
    if not candidate_tokens or not reference_tokens:
        return 0.0
    common = candidate_tokens & reference_tokens
    precision = len(common) / len(candidate_tokens)
    recall = len(common) / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_benchmark(benchmark_path: str = "data/benchmark.json") -> int:
    """Run all questions in benchmark.json through all 3 chunker types and score them."""
    path = Path(benchmark_path)
    if not path.exists():
        print(f"Benchmark file not found: {benchmark_path}")
        return 1

    with path.open(encoding="utf-8") as f:
        benchmarks: list[dict] = json.load(f)

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    # Build embedder once – reused for both indexing and scoring
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print("=" * 70)
    print("BENCHMARK EVALUATION – 3 CHUNKER TYPES")
    print(f"Scoring method : cosine similarity (embedder: {getattr(embedder, '_backend_name', embedder.__class__.__name__)})")
    print("=" * 70)

    totals: dict[str, float] = {ct: 0.0 for ct in CHUNKER_TYPES}

    for idx, item in enumerate(benchmarks, start=1):
        question: str = item["question"]
        best_answer: str = item["answer"]

        print(f"\n[Q{idx}] {question}")
        print(f"  Best answer : {best_answer}")
        print()

        for chunker_type in CHUNKER_TYPES:
            docs = load_documents_from_files(SAMPLE_FILES, chunker_type=chunker_type)
            if not docs:
                print(f"  [{chunker_type:>9}] No documents loaded – skipping.")
                continue

            store = EmbeddingStore(
                collection_name=f"benchmark_{chunker_type}",
                embedding_fn=embedder,
            )
            store.add_documents(docs)

            agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
            candidate = agent.answer(question, top_k=3)
            score = score_answer(candidate, best_answer, embedder=embedder)
            totals[chunker_type] += score

            print(f"  [{chunker_type:>9}] score={score:.3f}")
            print(f"             answer : {candidate}")
            print()

        print("-" * 70)

    n = len(benchmarks)
    print("\nAVERAGE SCORES ACROSS ALL QUESTIONS:")
    for ct in CHUNKER_TYPES:
        avg = totals[ct] / n if n else 0.0
        print(f"  {ct:>9} : {avg:.3f}")
    print("=" * 70)
    return 0


def main() -> int:
    args = sys.argv[1:]
    chunker_type = "recursive"
    filtered: list[str] = []
    i = 0
    while i < len(args):
        if args[i].startswith("--chunker="):
            chunker_type = args[i].split("=", 1)[1]
        elif args[i] == "--chunker" and i + 1 < len(args):
            chunker_type = args[i + 1]
            i += 1
        else:
            filtered.append(args[i])
        i += 1
    question = " ".join(filtered).strip() if filtered else None

    if question:
        return run_manual_demo(question=question, chunker_type=chunker_type)
    else:
        return run_benchmark()


if __name__ == "__main__":
    raise SystemExit(main())
