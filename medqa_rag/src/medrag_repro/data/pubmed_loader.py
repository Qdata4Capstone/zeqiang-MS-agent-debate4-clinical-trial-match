from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from datasets import load_dataset

from medrag_repro.datamodels import CorpusDoc
from medrag_repro.utils.text import normalize_ws


def iter_pubmed_rows(dataset_name: str = "ncbi/pubmed", streaming: bool = True) -> Iterator[Dict[str, Any]]:
    last_err: Optional[Exception] = None
    candidates = [dataset_name, "ncbi/pubmed", "MedRAG/pubmed"]
    for name in candidates:
        try:
            ds = load_dataset(name, streaming=streaming)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            for row in ds[split_name]:
                yield {"__dataset_name__": name, **row}
            return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load PubMed dataset. Last error: {last_err}")


def extract_pubmed_doc(row: Dict[str, Any], idx: int) -> Optional[CorpusDoc]:
    dataset_name = row.get("__dataset_name__", "unknown")
    title = ""
    abstract = ""
    doc_id = str(row.get("pmid") or row.get("id") or row.get("uid") or f"doc_{idx}")

    if "abstract" in row:
        abstract = normalize_ws(str(row.get("abstract") or ""))
    if "title" in row:
        title = normalize_ws(str(row.get("title") or ""))

    if not abstract and "content" in row:
        abstract = normalize_ws(str(row.get("content") or ""))
    if not title and "article_title" in row:
        title = normalize_ws(str(row.get("article_title") or ""))

    if not abstract and isinstance(row.get("MedlineCitation"), dict):
        mc = row["MedlineCitation"]
        article = mc.get("Article", {}) if isinstance(mc, dict) else {}
        title = normalize_ws(str(article.get("ArticleTitle") or title))
        ab = article.get("Abstract", {}) if isinstance(article, dict) else {}
        if isinstance(ab, dict):
            texts = ab.get("AbstractText")
            if isinstance(texts, list):
                abstract = normalize_ws(" ".join(str(x) for x in texts))
            elif texts is not None:
                abstract = normalize_ws(str(texts))

    if not abstract:
        return None

    text = normalize_ws((title + " " + abstract).strip())
    return CorpusDoc(doc_id=doc_id, title=title, abstract=abstract, text=text, source=dataset_name)
