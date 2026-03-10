import asyncio
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

from backend.agents.scoring import score_researcher


# ---------------------------------------------------------------------------
# Batching helper
# ---------------------------------------------------------------------------
async def batch_calls(
    items: list,
    func,
    batch_size: int = 2,
    delay: float = 2.0,
) -> list:
    results: list = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[func(item) for item in batch], return_exceptions=True
        )
        results.extend(batch_results)
        if i + batch_size < len(items):
            await asyncio.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# arXiv helpers
# ---------------------------------------------------------------------------
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _parse_arxiv_entries(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    entries: List[Dict[str, Any]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        title_el = entry.find("atom:title", ARXIV_NS)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        authors: List[str] = []
        for author_el in entry.findall("atom:author", ARXIV_NS):
            name_el = author_el.find("atom:name", ARXIV_NS)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
        abstract_el = entry.find("atom:summary", ARXIV_NS)
        abstract = abstract_el.text.strip().replace("\n", " ") if abstract_el is not None and abstract_el.text else ""
        entries.append({"title": title, "authors": authors, "abstract": abstract})
    return entries


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------
async def _s2_search_author(name: str, client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/author/search"
    resp = await client.get(url, params={"query": name, "limit": 3}, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


async def _s2_get_author(author_id: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    fields = (
        "name,hIndex,citationCount,paperCount,"
        "papers.title,papers.year,papers.citationCount,"
        "papers.externalIds,papers.abstract"
    )
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}"
    resp = await client.get(url, params={"fields": fields}, timeout=30.0)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Theme extraction
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with",
    "from", "by", "is", "are", "was", "were", "at", "as", "it", "its",
    "via", "using", "based", "towards", "through", "into",
}


def _extract_themes(paper_titles: List[str], top_n: int = 5) -> List[str]:
    word_counts: Dict[str, int] = {}
    for title in paper_titles:
        words = re.findall(r"[a-z]{3,}", title.lower())
        for w in words:
            if w not in _STOPWORDS:
                word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


# ---------------------------------------------------------------------------
# Build lightweight profile
# ---------------------------------------------------------------------------
def _build_light_profile(author_data: Dict[str, Any]) -> Dict[str, Any]:
    papers = author_data.get("papers") or []
    papers_sorted = sorted(papers, key=lambda p: (p.get("citationCount") or 0), reverse=True)

    top_paper: Optional[Dict[str, Any]] = None
    if papers_sorted:
        p = papers_sorted[0]
        top_paper = {
            "title": p.get("title", ""),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
        }

    titles = [p.get("title", "") for p in papers_sorted[:20]]
    abstracts = [p.get("abstract", "") or "" for p in papers_sorted[:20]]
    themes = _extract_themes(titles)
    relevance = score_researcher(titles, themes, abstracts)

    author_id = author_data.get("authorId", "")
    return {
        "name": author_data.get("name", ""),
        "citations": author_data.get("citationCount", 0),
        "hIndex": author_data.get("hIndex", 0),
        "paperCount": author_data.get("paperCount", 0),
        "topPaper": top_paper,
        "relevanceScore": relevance,
        "themes": themes,
        "semanticScholarUrl": f"https://www.semanticscholar.org/author/{author_id}",
    }


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------
async def find_emerging_researchers(topic: str) -> List[Dict[str, Any]]:
    """Find emerging (low-citation) researchers working on a topic."""

    async with httpx.AsyncClient(follow_redirects=True) as client:
        # 1. Search arXiv for very recent papers
        arxiv_url = (
            f"https://export.arxiv.org/api/query?"
            f'search_query=all:"{topic}"+AND+'
            f"(cat:cs.CV+OR+cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.GR+OR+cat:cs.RO)"
            f"&max_results=30&sortBy=submittedDate"
        )
        resp = await client.get(arxiv_url, timeout=30.0)
        resp.raise_for_status()
        entries = _parse_arxiv_entries(resp.text)

        # 2. Extract ALL authors (not just first authors)
        unique_authors: List[str] = []
        seen_lower: set = set()
        for entry in entries:
            for name in entry["authors"]:
                if name.lower() not in seen_lower:
                    seen_lower.add(name.lower())
                    unique_authors.append(name)

        # Cap to avoid too many API calls
        unique_authors = unique_authors[:20]

        if not unique_authors:
            return []

        # 3. Search S2 for each author (batched)
        async def _search_s2(name: str) -> List[Dict[str, Any]]:
            try:
                return await _s2_search_author(name, client)
            except Exception:
                return []

        s2_search_results = await batch_calls(unique_authors, _search_s2, batch_size=2, delay=2.0)

        # Pick first S2 match per name, collect author IDs
        author_ids: List[str] = []
        seen_ids: set = set()
        for result_list in s2_search_results:
            if not isinstance(result_list, list) or not result_list:
                continue
            aid = result_list[0].get("authorId")
            if aid and aid not in seen_ids:
                seen_ids.add(aid)
                author_ids.append(aid)

        if not author_ids:
            return []

        # 4. Get full profiles (batched)
        async def _get_s2(aid: str) -> Optional[Dict[str, Any]]:
            try:
                return await _s2_get_author(aid, client)
            except Exception:
                return None

        full_authors = await batch_calls(author_ids, _get_s2, batch_size=2, delay=2.0)

        # 5. Filter to emerging researchers (< 5000 citations)
        profiles: List[Dict[str, Any]] = []
        for author_data in full_authors:
            if not isinstance(author_data, dict) or not author_data:
                continue
            citation_count = author_data.get("citationCount", 0) or 0
            if citation_count < 5000:
                profiles.append(_build_light_profile(author_data))

        # 6. Sort by relevance score descending
        profiles.sort(key=lambda p: p.get("relevanceScore", 0), reverse=True)

    return profiles
