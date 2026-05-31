"""End-to-end orchestrator: synonyms + categories -> deduplicated papers.json."""

import json
import logging
from pathlib import Path
from typing import Iterable, List

from .boolean import build_boolean_queries, load_synonyms
from .config import PaperDiscoveryConfig
from .pdf import download_pdf, extract_text
from .schema import CategoryQuery, Paper
from .sources import get_adapter

logger = logging.getLogger(__name__)


class PaperDiscoveryPipeline:
    def __init__(self, config: PaperDiscoveryConfig):
        self.config = config

    def run(self) -> List[Paper]:
        cfg = self.config
        synonyms = load_synonyms(cfg.synonyms_path)
        queries = build_boolean_queries(synonyms, cfg.categories)
        logger.info("Built %d category queries", len(queries))

        all_papers: List[Paper] = []
        for q in queries:
            all_papers.extend(self._fetch_one(q))

        deduped = _dedupe(all_papers)
        logger.info("After dedupe: %d papers (from %d)", len(deduped), len(all_papers))

        if cfg.download_pdfs:
            self._enrich_with_pdf_text(deduped)

        self._write_output(deduped)
        return deduped

    def _fetch_one(self, query: CategoryQuery) -> List[Paper]:
        cfg = self.config
        out: List[Paper] = []
        for src_name in cfg.sources:
            adapter = get_adapter(
                src_name,
                user_agent=cfg.user_agent,
                max_pages=cfg.max_pages,
                max_results=cfg.max_results,
                **(
                    {"category_map": cfg.arxiv_category_map}
                    if src_name == "arxiv" and cfg.arxiv_category_map
                    else {}
                ),
            )
            logger.info("Searching %s for %r", src_name, query.combination_title)
            papers = adapter.search(query.boolean_combination, query.combination_title)
            logger.info("%s returned %d papers", src_name, len(papers))
            out.extend(papers)
        return out

    def _enrich_with_pdf_text(self, papers: Iterable[Paper]) -> None:
        cfg = self.config
        for paper in papers:
            if not paper.url:
                continue
            path = download_pdf(paper.url, cfg.pdf_dir, user_agent=cfg.user_agent)
            if not path:
                continue
            paper.extracted_text = extract_text(path) or None

    def _write_output(self, papers: List[Paper]) -> None:
        out_path = Path(self.config.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [p.to_dict() for p in papers]
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Wrote %d papers to %s", len(papers), out_path)


def _dedupe(papers: List[Paper]) -> List[Paper]:
    seen = set()
    out: List[Paper] = []
    for p in papers:
        key = (p.title or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
