"""Lightweight local RAG using Chroma with on-disk persistence.

Collections:
- code: chunks from repository code (Python files under project root/edbo by default)
- papers: optional external papers (not indexed by default here)
- csv: per-column docs for the active dataset (scoped by dataset path)

Embeddings: sentence-transformers (all-MiniLM-L6-v2) for free, local embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple
import hashlib
import os
import json

import chromadb  # type: ignore
from chromadb import Collection  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore


# ---------------- Embeddings ---------------- #

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Embedder:
    model_name: str = _EMBED_MODEL_NAME
    _model: Optional[SentenceTransformer] = None

    def _ensure(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure()
        assert self._model is not None
        return self._model.encode(texts, normalize_embeddings=True).tolist()


# ---------------- Chroma client and collections ---------------- #

def get_client(persist_dir: Path):
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collections(client, names: Optional[List[str]] = None) -> Dict[str, Collection]:
    """Return a dict of named collections, creating if missing.

    names defaults to ("code", "papers", "csv") for convenience, but callers
    can pass any list they need. This avoids hard-coding specific use cases.
    """
    if names is None:
        names = ["code", "papers", "csv"]
    cols: Dict[str, Collection] = {}
    for name in names:
        cols[name] = client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return cols


# ---------------- Utilities ---------------- #

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_text(text: str, chunk_chars: int = 1800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        chunks.append(text[i:j])
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


# ---------------- Ingestion ---------------- #

def ingest_codebase(
    client,
    embedder: Embedder,
    project_root: Path,
    include_dirs: Optional[List[Path]] = None,
    file_glob: str = "*.py",
    exclude_contains: Optional[List[str]] = None,
    reindex: bool = False,
) -> int:
    """Index code files as chunks into 'code' collection. Returns upserted count.

    - include_dirs: directories to scan; defaults to [project_root] when None
    - file_glob: filename pattern (e.g., "*.py")
    - exclude_contains: skip paths containing any of these substrings
    - reindex: force re-embedding even if hash sentinel exists
    """
    cols = get_collections(client)
    code = cols["code"]
    if include_dirs is None:
        include_dirs = [project_root]
    if exclude_contains is None:
        exclude_contains = ["__pycache__"]
    upserted = 0
    for d in include_dirs:
        if not d.exists():
            continue
        for path in d.rglob(file_glob):
            spath = str(path)
            if any(s in spath for s in (exclude_contains or [])):
                continue
            rel = path.relative_to(project_root)
            text = _read_text(path)
            doc_hash = _sha256(str(rel) + "::" + _sha256(text))
            # If existing and not reindex, skip
            if not reindex:
                # Chroma has no direct get by where+limit=1 without filters; use count check via query on id prefix
                # We'll use deterministic chunk ids and check presence of a sentinel id
                sentinel_id = f"code:{rel.as_posix()}:0:{doc_hash[:12]}"
                try:
                    _ = code.get(ids=[sentinel_id])
                    # If present, assume file is already indexed for this hash
                    if _ and _["ids"]:
                        continue
                except Exception:
                    pass
            chunks = chunk_text(text)
            if not chunks:
                continue
            ids = [f"code:{rel.as_posix()}:{idx}:{doc_hash[:12]}" for idx, _ in enumerate(chunks)]
            metadatas = [{"path": rel.as_posix(), "type": "code", "idx": idx, "hash": doc_hash} for idx, _ in enumerate(chunks)]
            embeddings = embedder.embed(chunks)
            code.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
            upserted += len(chunks)
    return upserted


def _read_notebook_cells(nb_path: Path) -> List[Tuple[int, str, str, str]]:
    """Return list of (cell_idx, cell_type, language, text) from an .ipynb file.
    Falls back gracefully if fields are missing.
    """
    try:
        data = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    cells = data.get("cells", [])
    out: List[Tuple[int, str, str, str]] = []
    for idx, cell in enumerate(cells, start=1):
        ctype = cell.get("cell_type", "unknown")
        meta = cell.get("metadata", {}) or {}
        lang = (meta.get("language") or meta.get("kernelspec", {}).get("language") or
                (data.get("metadata", {}).get("language_info", {}).get("name"))) or ""
        src_list = cell.get("source", []) or []
        # source may be a list or string
        if isinstance(src_list, list):
            text = "".join(src_list)
        else:
            text = str(src_list)
        # Skip empty cells
        if not text.strip():
            continue
        out.append((idx, ctype, str(lang), text))
    return out


def ingest_ipynb(client, embedder: Embedder, project_root: Path, nb_path: Path, reindex: bool = False) -> int:
    """Index notebook cells into 'papers' collection. Returns upserted count."""
    cols = get_collections(client)
    papers = cols["papers"]
    if not nb_path.exists():
        return 0
    rel = nb_path.relative_to(project_root) if str(nb_path).startswith(str(project_root)) else nb_path
    cells = _read_notebook_cells(nb_path)
    upserted = 0
    for (cell_idx, ctype, lang, text) in cells:
        base = f"nb:{rel.as_posix()}:{cell_idx}:{ctype}:{lang}"
        doc_hash = _sha256(base + "::" + _sha256(text))
        _id = f"nb:{rel.as_posix()}:{cell_idx}:{doc_hash[:12]}"
        if not reindex:
            try:
                existing = papers.get(ids=[_id])
                if existing and existing.get("ids"):
                    continue
            except Exception:
                pass
        # Build a short header with context
        header = f"Notebook: {rel.as_posix()}\nCell {cell_idx} ({ctype}, {lang})\n"
        doc = header + text
        chunks = chunk_text(doc)
        ids = []
        documents = []
        metadatas = []
        for i, ch in enumerate(chunks):
            ids.append(f"{_id}:{i}")
            documents.append(ch)
            metadatas.append({
                "type": "notebook",
                "path": rel.as_posix(),
                "cell_index": cell_idx,
                "cell_type": ctype,
                "language": lang,
                "hash": doc_hash,
            })
        if ids:
            embeddings = embedder.embed(documents)
            papers.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            upserted += len(ids)
    return upserted


def build_csv_column_docs(df, descriptor_lookup) -> List[Tuple[str, str]]:
    """Return list of (title, content) doc strings for per-column descriptions."""
    docs: List[Tuple[str, str]] = []
    for col in df.columns:
        series = df[col]
        values_sample = ", ".join(map(str, series.dropna().unique().tolist()[:20]))
        desc = None
        if descriptor_lookup:
            try:
                desc = descriptor_lookup.get(col)
            except Exception:
                desc = None
        stats = ""
        try:
            if series.dtype.kind in ("i", "f"):
                stats = f"; min={series.min()}, max={series.max()}, mean={series.mean():.4f}"
        except Exception:
            pass
        body = f"Column: {col}\nDefinition: {desc or '(unknown)'}\nSample values: {values_sample}{stats}"
        docs.append((col, body))
    return docs


def ingest_csv_columns(client, embedder: Embedder, df, dataset_path: Path, descriptor_lookup=None) -> int:
    cols = get_collections(client)
    csv_col = cols["csv"]
    upserted = 0
    docs = build_csv_column_docs(df, descriptor_lookup)
    dataset_key = str(dataset_path.resolve())
    ids = []
    documents = []
    metadatas = []
    for title, content in docs:
        doc_hash = _sha256(dataset_key + "::" + title + "::" + _sha256(content))
        _id = f"csv:{dataset_key}:{title}:{doc_hash[:12]}"
        # Skip if doc with same id exists
        try:
            existing = csv_col.get(ids=[_id])
            if existing and existing["ids"]:
                continue
        except Exception:
            pass
        ids.append(_id)
        documents.append(content)
        metadatas.append({"dataset": dataset_key, "column": title, "type": "csv", "hash": doc_hash})
    if ids:
        embeddings = embedder.embed(documents)
        csv_col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        upserted = len(ids)
    return upserted


# ---------------- Retrieval ---------------- #

def query(client, embedder: Embedder, query_text: str, top_k: int = 4, collections: Optional[List[str]] = None, where: Optional[Dict] = None) -> List[Dict]:
    if collections is None:
        collections = ["code", "csv", "papers"]
    all_hits: List[Dict] = []
    q_emb = embedder.embed([query_text])[0]
    for name in collections:
        col = get_collections(client)[name]
        try:
            res = col.query(query_embeddings=[q_emb], n_results=top_k, where=where)
            # Normalize results
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0] or []
            for i, doc in enumerate(docs):
                all_hits.append({
                    "collection": name,
                    "id": ids[i] if i < len(ids) else None,
                    "document": doc,
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                })
        except Exception:
            continue
    # Sort by distance (cosine; lower is better if distances provided) else stable
    all_hits.sort(key=lambda x: (x["distance"] if x["distance"] is not None else 1e9))
    return all_hits[:top_k]


def format_hits_for_prompt(hits: List[Dict], max_chars: int = 2000) -> str:
    lines = []
    for h in hits:
        meta = h.get("metadata", {})
        src = meta.get("path") or meta.get("dataset") or meta.get("column") or h.get("collection")
        lines.append(f"[{h['collection']}] {src}:\n{h['document'][:600].strip()}")
    text = "\n---\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + "\n... (truncated)"
    return text
